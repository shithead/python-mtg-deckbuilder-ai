import numpy as np

from .Pool import Pool

class Sideboard(Pool):
    def __init__(self, deck_size : int ):
        super().__init__(self)
        self._size = 15
        self._oversize = int(4/ deck_size)
        self.body = np.zeros(shape=(self.__size))
        self.oversize = np.zeros(shape=(self._oversize))

