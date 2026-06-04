import random
import torch
from torch.utils.data import Dataset
from database.mtgtools import Database
from ai.CardEncoder import CardEncoder


class DeckDataset(Dataset):
    def __init__(self, db: Database, encoder: CardEncoder, num_negatives: int = 1):
        self._decks = list(db.root.wcc_decks)
        self._encoder = encoder
        self._num_negatives = num_negatives

        all_cards = db.loadPool().unique_names() + db.loadWccPool().unique_names()
        self._all_card_names = [c.name for c in all_cards]
        self._all_embeddings = encoder.encode_many(all_cards)

        self._deck_card_pool_indices = []
        for deck in self._decks:
            indices = []
            for card in deck:
                for pi, pool_card in enumerate(all_cards):
                    if pool_card.name.lower() == card.name.lower():
                        indices.append(pi)
                        break
            self._deck_card_pool_indices.append(indices)

        self._samples = []
        for deck_idx, deck in enumerate(self._decks):
            pool_indices = self._deck_card_pool_indices[deck_idx]

            deck_card_names = {c.name.lower() for c in deck}
            neg_candidates = [
                pi for pi, name in enumerate(self._all_card_names)
                if name.lower() not in deck_card_names
            ]

            for pos in range(len(pool_indices)):
                self._samples.append((deck_idx, pos, 1.0))
                for _ in range(num_negatives):
                    neg_pool_idx = random.choice(neg_candidates)
                    self._samples.append((deck_idx, neg_pool_idx, 0.0))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        deck_idx, ref, label = self._samples[idx]

        if label == 1.0:
            card_pos = ref
            pool_indices = self._deck_card_pool_indices[deck_idx]
            deck_embs = self._all_embeddings[pool_indices]

            mask = torch.ones(len(pool_indices), dtype=torch.bool)
            mask[card_pos] = False
            deck_ctx = deck_embs[mask].mean(dim=0)

            card_emb = self._all_embeddings[pool_indices[card_pos]]
        else:
            neg_pool_idx = ref
            pool_indices = self._deck_card_pool_indices[deck_idx]
            deck_embs = self._all_embeddings[pool_indices]

            deck_ctx = deck_embs.mean(dim=0)

            card_emb = self._all_embeddings[neg_pool_idx]

        x = torch.cat([deck_ctx, card_emb])
        y = torch.tensor([label], dtype=torch.float32)
        return x, y
