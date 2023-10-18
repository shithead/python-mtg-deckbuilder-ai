
from .Card import MTGCard
from .Pool import Pool
from .Deck import Deck

class Constructor():
    def __init__(self):
        pass

    def other_card(self, pool: Pool, action: int = 0):
        '''
        action:
            no action - 0
            preview card - 1
            next card - 2

        show card from pool .
        '''
        if action == 1:
            return pool.prev
        if action == 2:
            return pool.next

    def this_card(self, pool: Pool, deck: Deck ,action: int = 0):
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

