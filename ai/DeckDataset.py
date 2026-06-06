import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from database.mtgtools import Database
from ai.CardEncoder import CardEncoder


class DeckDataset(Dataset):
    def __init__(self, db: Database, encoder: CardEncoder, num_negatives: int = 1,
                 num_synthetic: int = 0, hard_negatives: int = 0):
        self._decks = list(db.root.wcc_decks)
        self._encoder = encoder
        self._num_negatives = num_negatives

        all_cards = db.loadPool().unique_names() + db.loadWccPool().unique_names()
        self._all_card_names = [c.name for c in all_cards]
        self._all_embeddings = encoder.encode_many(all_cards)

        norm = self._all_embeddings.norm(dim=1, keepdim=True)
        norm[norm == 0] = 1
        all_embs_norm = self._all_embeddings / norm

        self._name_to_idx = {c.name.lower(): i for i, c in enumerate(all_cards)}
        self._deck_card_pool_indices = [
            [self._name_to_idx[card.name.lower()] for card in deck]
            for deck in self._decks
        ]

        self._samples = []
        for deck_idx, deck in enumerate(self._decks):
            pool_indices = self._deck_card_pool_indices[deck_idx]
            deck_idx_set = set(pool_indices)

            deck_card_names = {c.name.lower() for c in deck}
            neg_candidates = [
                pi for pi, name in enumerate(self._all_card_names)
                if name.lower() not in deck_card_names
            ]

            deck_emb = all_embs_norm[pool_indices].mean(dim=0, keepdim=True)
            sims = all_embs_norm @ deck_emb.T
            sims.squeeze_(-1)

            if hard_negatives > 0 and len(neg_candidates) > hard_negatives:
                sims_for_neg = sims.clone()
                sims_for_neg[list(deck_idx_set)] = -1
                topk = torch.topk(sims_for_neg, min(len(neg_candidates), hard_negatives))
                hard_pool = topk.indices.tolist()
            else:
                hard_pool = neg_candidates

            for pos in range(len(pool_indices)):
                self._samples.append((deck_idx, pool_indices[pos], 1.0, False))
                for _ in range(num_negatives):
                    neg_pool_idx = random.choice(hard_pool if hard_negatives > 0 else neg_candidates)
                    self._samples.append((deck_idx, neg_pool_idx, 0.0, False))

            if num_synthetic > 0:
                sims[list(deck_idx_set)] = -1
                topk = torch.topk(sims, num_synthetic)
                for syn_idx in topk.indices.tolist():
                    self._samples.append((deck_idx, syn_idx, 1.0, True))

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
            deck_ctx = deck_embs[mask].mean(dim=0)
            card_emb = self._all_embeddings[card_idx]
        else:
            deck_ctx = deck_embs.mean(dim=0)
            card_emb = self._all_embeddings[card_idx]

        x = torch.cat([deck_ctx, card_emb])
        y = torch.tensor([label], dtype=torch.float32)
        return x, y
