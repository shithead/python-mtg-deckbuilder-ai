from mtgtools.PCard import PCard
from mtgtools.PCardList import PCardList
from .Card import AICard

class Deck(PCardList):
    def __init__(self, maxsize : int = 60):
        super().__init__()
        self.__maxsize = maxsize
        self.__minsize = 60

    def update_deck(self, card : PCard = None, action: int = 0):
        '''
        action:
            no action = 0
            remove = 3
            add = 4
        '''
        if action == 4:
            if len(self) < self.__maxsize:
                self.append(card)
            else:
                raise ValueError("Can not add Card to Deck, max size reached.")
        if action == 3:
            self.pop(-1)
