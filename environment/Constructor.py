
from mtgtools.PCardList import PCardList
from .Card import AICard
from .Deck import Deck

class Constructor():
    def __init__(self):
        self.__pos_pool = 0
        self.__searcher = None

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
            card = deck.current
            deck.update_deck(action = action)
            pool.add(card)
        if action == 4:
            card = pool.current
            del pool.current
            deck.update_deck(card = card, action = action)

