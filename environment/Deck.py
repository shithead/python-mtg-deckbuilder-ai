import numpy as np

from .Card import Card, TCard
from .Sideboard import Sideboard as SB
from .Pool import Pool

class Deck(Pool):
    def __init__(self, maxsize : int = 60, mana_colors: list = [1, 1, 1, 1, 1, 1]):
        super().__init__(self)
        self.__maxsize = maxsize
        self.__minsize = 60
        self.mana_colors = mana_colors
        self.body = np.zeros(shape=(self.size))
        self.sideboard = SB(self.size)
        self.__legal: list = None

    def update_deck(self, card : TCard = None, action: int = 0):
        '''
        action:
            no action = 0
            remove = 4
            add = 3
        '''
        if action == 3:
            if self.size <= self.__maxsize:
                self.add(card)
            else:
                raise BufferError("Can not delete Card from Deck.")
        if action == 4:
            del self.current 
