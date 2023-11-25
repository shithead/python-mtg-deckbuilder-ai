
from mtgtools.PCardList import PCardList
from .Card import AICard
from .Deck import Deck

class Constructor():
    def __init__(self):
        self.__pos_pool = 0
        pass

    def other_card(self, pool: PCardList, action: int = 0):
        '''
        action:
            no action - 0
            preview card - 1
            next card - 2

        show card from pool .
        '''
        if action == 1:
            self.__pos_pool += 1
        if action == 2:
            self.__pos_pool -= 1
        return pool[self.__pos_pool]

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
            del pool.currnet
            deck.update_deck(card = card, action = action)

