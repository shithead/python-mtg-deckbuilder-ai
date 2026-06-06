
from mtgtools.PCard import PCard
from mtgtools.PCardList import PCardList
from .Card import AICard
from .Deck import Deck
from config import BASIC_LAND_TYPES, MAX_COPIES, SYNERGY_MODEL_PATH

class Constructor():
    def __init__(self):
        self.__pos_pool = 0
        self.__searcher = None
        self.__encoder = None
        self.__model = None

    def _get_searcher(self):
        if self.__searcher is None:
            from database.VectorDB import VectorSearcher
            self.__searcher = VectorSearcher()
        return self.__searcher

    def suggest(self, pool: PCardList, query_text: str, n_results: int = 5):
        """
        Return a card from the pool that matches the query text
        using semantic vector search, or None if no match is found.
        """
        if len(pool) == 0:
            return None
        searcher = self._get_searcher()
        indices = searcher.suggest_indices(query_text, n_results)
        for idx in indices:
            if idx < len(pool):
                return pool[idx]
        return None

    def other_card(self, pool: PCardList, action: int = 0):
        '''
        action:
            no action - 0
            next card - 1
            prev card - 2

        Return current card from pool, then move position.
        '''
        if len(pool) == 0:
            return None
        self.__pos_pool = max(0, min(self.__pos_pool, len(pool) - 1))
        card = pool[self.__pos_pool]
        if action == 1:
            self.__pos_pool = min(len(pool) - 1, self.__pos_pool + 1)
        if action == 2:
            self.__pos_pool = max(0, self.__pos_pool - 1)
        return card

    def this_card(self, pool: PCardList, deck: Deck ,action: int = 0):
        '''
        action:
            no action - 0
            drop - 3
            pick - 4

        Pick and remove card from pool and add to deck.
        Drop card from deck and add back to pool.
        '''
        if action == 3:
            card = deck[-1]
            deck.update_deck(action = action)
            pool.append(card)
        if action == 4:
            card = pool[-1]
            pool.pop(-1)
            deck.update_deck(card = card, action = action)

    def can_add_copy(self, deck: Deck, card: PCard) -> bool:
        type_text = getattr(card, "type_line", None) or getattr(card, "type", "") or ""
        if "Basic" in type_text:
            return True
        count = sum(1 for c in deck if c.name == card.name)
        return count < MAX_COPIES

    def add_card_to_deck(self, pool: PCardList, deck: Deck, card: PCard) -> bool:
        if not self.can_add_copy(deck, card):
            return False
        if card in pool:
            pool.remove(card)
            deck.append(card)
            return True
        return False

    def _get_scorer(self):
        if self.__model is None:
            import torch
            from ai.CardEncoder import CardEncoder
            from ai.SynergyClassifier import SynergyClassifier
            self.__encoder = CardEncoder()
            self.__model = SynergyClassifier()
            self.__model.load_state_dict(torch.load(SYNERGY_MODEL_PATH, map_location="cpu"))
            self.__model.eval()
        return self.__model, self.__encoder

    def score_card(self, deck: Deck, card: PCard, alpha: float = 1.0) -> float:
        if len(deck) == 0:
            return 0.5
        model, encoder = self._get_scorer()
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            deck_embs = encoder.encode_many(list(deck))
            deck_ctx = deck_embs.mean(dim=0)
            card_emb = encoder.encode(card)
            x = torch.cat([deck_ctx, card_emb]).unsqueeze(0)
            score = model(x).item()
        if alpha < 1.0:
            cos_sim = F.cosine_similarity(deck_ctx.unsqueeze(0), card_emb.unsqueeze(0)).item()
            cos_sim = (cos_sim + 1) / 2
            score = alpha * score + (1 - alpha) * cos_sim
        return score

    def rank_cards(self, deck: Deck, cards, alpha: float = 1.0) -> list:
        if len(deck) == 0:
            return list(cards)
        model, encoder = self._get_scorer()
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            deck_embs = encoder.encode_many(list(deck))
            deck_ctx = deck_embs.mean(dim=0)
            card_embs = encoder.encode_many(list(cards))
            deck_ctx_exp = deck_ctx.unsqueeze(0).expand(len(cards), -1)
            x = torch.cat([deck_ctx_exp, card_embs], dim=1)
            model_scores = model(x).squeeze(-1)
        if alpha < 1.0:
            cos_sim = F.cosine_similarity(deck_ctx.unsqueeze(0), card_embs, dim=1)
            cos_sim = (cos_sim + 1) / 2
            scores = (alpha * model_scores + (1 - alpha) * cos_sim).tolist()
        else:
            scores = model_scores.tolist()
        return sorted(zip(cards, scores), key=lambda cs: cs[1], reverse=True)
