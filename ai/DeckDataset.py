import random
from collections.abc import Mapping
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from database.mtgtools import Database
from ai.CardEncoder import CardEncoder


BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


def _is_basic_land(card) -> bool:
    type_text = (getattr(card, "type_line", None) or getattr(card, "type", "") or "")
    return "basic" in type_text.lower()


def _card_legal(card, fmt) -> bool:
    """Best-effort format legality check (Scryfall dict or mtgio list)."""
    if fmt is None:
        return True
    legalities = getattr(card, "legalities", None)
    fmt_l = str(fmt).lower()
    if isinstance(legalities, dict):
        val = legalities.get(fmt_l) or legalities.get(str(fmt))
        return str(val).lower() == "legal"
    if isinstance(legalities, list):
        for leg in legalities:
            if isinstance(leg, dict):
                if str(leg.get("format", "")).lower() == fmt_l:
                    return str(leg.get("legality", "")).lower() == "legal"
    return True


class DeckDataset(Dataset):
    def __init__(self, db: Database, encoder: CardEncoder, num_negatives: int = 1,
                 num_synthetic: int = 0, hard_negatives: int = 0,
                 sources=None, formats=None, balance: bool = True,
                 exclude_basic_lands: bool = True):
        self._encoder = encoder
        self._num_negatives = num_negatives
        self._num_synthetic = num_synthetic
        self._hard_negatives = hard_negatives
        self._exclude_basic_lands = exclude_basic_lands

        # --- Resolve deck entries (deck, source, format) ---------------
        entries = self._resolve_entries(db, sources, formats, balance)
        self._deck_entries = entries
        self._decks = [e[0] for e in entries]
        self._deck_sources = [e[1] for e in entries]
        self._deck_formats = [e[2] for e in entries]

        # --- Pool of all cards ---------------------------------------
        try:
            all_cards = db.loadPool().unique_names() + db.loadWccPool().unique_names()
        except Exception:
            all_cards = []
        self._all_card_names = [c.name for c in all_cards]
        self._all_embeddings = encoder.encode_many(all_cards)

        norm = self._all_embeddings.norm(dim=1, keepdim=True)
        norm[norm == 0] = 1
        self._all_embs_norm = self._all_embeddings / norm

        self._name_to_idx = {c.name.lower(): i for i, c in enumerate(all_cards)}

        self._deck_card_pool_indices = []
        for deck in self._decks:
            indices = []
            for card in deck:
                idx = self._name_to_idx.get(card.name.lower())
                if idx is not None:
                    indices.append(idx)
            self._deck_card_pool_indices.append(indices)

        # --- Build samples -------------------------------------------
        self._samples = []
        for deck_idx, (deck, source, fmt) in enumerate(self._deck_entries):
            pool_indices = self._deck_card_pool_indices[deck_idx]
            deck_idx_set = set(pool_indices)

            deck_card_names = {c.name.lower() for c in deck}
            all_neg_candidates = [
                pi for pi, name in enumerate(self._all_card_names)
                if name.lower() not in deck_card_names
            ]

            # Format-aware negatives
            fmt_neg_candidates = [
                pi for pi in all_neg_candidates
                if _card_legal(all_cards[pi], fmt)
            ] or all_neg_candidates

            deck_emb = self._all_embs_norm[pool_indices].mean(dim=0, keepdim=True) if pool_indices else None
            if deck_emb is not None:
                sims = self._all_embs_norm @ deck_emb.T
                sims.squeeze_(-1)
            else:
                sims = None

            # Hard negatives
            hard_pool = fmt_neg_candidates
            if self._hard_negatives > 0 and len(fmt_neg_candidates) > self._hard_negatives and sims is not None:
                sims_for_neg = sims.clone()
                sims_for_neg[list(deck_idx_set)] = -1
                # only consider legal candidates
                mask = torch.zeros(len(self._all_card_names), dtype=torch.bool)
                for pi in fmt_neg_candidates:
                    mask[pi] = True
                sims_for_neg[~mask] = -1
                topk = torch.topk(sims_for_neg, min(len(fmt_neg_candidates), self._hard_negatives))
                hard_pool = topk.indices.tolist()

            for pos_i, card in enumerate(deck):
                pool_idx = pool_indices[pos_i] if pos_i < len(pool_indices) else None
                if pool_idx is None:
                    continue
                if self._exclude_basic_lands and _is_basic_land(card):
                    continue
                self._samples.append((deck_idx, pool_idx, 1.0, False))

                for _ in range(self._num_negatives):
                    if self._hard_negatives > 0:
                        neg_pool_idx = random.choice(hard_pool)
                    else:
                        neg_pool_idx = random.choice(fmt_neg_candidates)
                    self._samples.append((deck_idx, neg_pool_idx, 0.0, False))

            if self._num_synthetic > 0 and sims is not None:
                sims[list(deck_idx_set)] = -1
                # exclude basic lands from synthetic positives
                if self._exclude_basic_lands:
                    for pi, name in enumerate(self._all_card_names):
                        if _is_basic_land(all_cards[pi]):
                            sims[pi] = -1
                topk = torch.topk(sims, min(self._num_synthetic, len(sims)))
                for syn_idx in topk.indices.tolist():
                    if syn_idx in deck_idx_set:
                        continue
                    self._samples.append((deck_idx, syn_idx, 1.0, True))

    def _resolve_entries(self, db, sources, formats, balance):
        collections = getattr(db.root, "deck_collections", None)
        formats_map = getattr(db.root, "deck_formats", None)

        if isinstance(collections, Mapping) and collections:
            if sources is None:
                sources = list(collections.keys())
            entries = []
            for src in sources:
                if src not in collections:
                    continue
                fmt_list = formats_map.get(src) if isinstance(formats_map, Mapping) else None
                for i, deck in enumerate(collections[src]):
                    fmt = None
                    if isinstance(fmt_list, (list, tuple)) and i < len(fmt_list):
                        fmt = fmt_list[i]
                    if formats is not None and fmt not in formats:
                        continue
                    entries.append((list(deck), src, fmt))

            # Balance across sources
            if balance and len({e[1] for e in entries}) > 1:
                by_source = {}
                for e in entries:
                    by_source.setdefault(e[1], []).append(e)
                min_pos = min(sum(len(e[0]) for e in ds) for ds in by_source.values())
                balanced = []
                for src, ds in by_source.items():
                    acc = 0
                    for e in ds:
                        if acc >= min_pos:
                            break
                        balanced.append(e)
                        acc += len(e[0])
                entries = balanced
            return entries

        # Backward compat: wcc_decks
        wcc = getattr(db.root, "wcc_decks", None)
        if wcc is None:
            wcc = []
        entries = []
        for deck in wcc:
            entries.append((list(deck), "wcc", None))
        if formats is not None:
            entries = [e for e in entries if e[2] in formats]
        return entries

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        deck_idx, card_idx, label, synthetic = self._samples[idx]
        pool_indices = self._deck_card_pool_indices[deck_idx]
        deck_embs = self._all_embeddings[pool_indices]

        if label == 1.0 and not synthetic:
            card_pos = pool_indices.index(card_idx)
            mask = torch.ones(len(pool_indices), dtype=torch.bool)
            mask[card_pos] = False
            deck_ctx = deck_embs[mask].mean(dim=0) if mask.any() else deck_embs.mean(dim=0)
            card_emb = self._all_embeddings[card_idx]
        else:
            deck_ctx = deck_embs.mean(dim=0) if len(pool_indices) else torch.zeros(self._all_embeddings.shape[1])
            card_emb = self._all_embeddings[card_idx]

        x = torch.cat([deck_ctx, card_emb])
        y = torch.tensor([label], dtype=torch.float32)
        return x, y