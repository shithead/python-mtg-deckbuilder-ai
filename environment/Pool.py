from .Card import Card, TCard

class Pool():
    def __init__(self):
        self._first : TCard = None
        self._last : TCard = None
        self._cur : TCard = None
        self.__size : int = 0

    @property
    def size(self) -> int:
        return self.__size

    @property
    def first(self) -> TCard:
        self._cur = self._first
        return self._first

    @first.setter
    def first(self, card: TCard):
        if card is None:
           raise AttributeError('card can not be NoneType') 
        if self._first is None:
            self._last = card
            card._next = card._prev = card
        else:
            self._last.next = card
            self._first.prev = card
        self._first = card
        self._cur = self._first

    @property
    def last(self) -> TCard:
        self._cur = self._last
        return self._last

    @last.setter
    def last(self, card: TCard):
        if card is None:
           raise AttributeError('card can not be NoneType') 
        if self._last is None:
            # if last is None than first is None too
            self._first = card
            card._next = card._prev = card
        else:
            self._last.next = card
            self._first.prev = card
        self._last = card
        self._cur = self._last

    @property
    def current(self) -> TCard:
        return self._cur

    @current.deleter
    def current(self):
        card = self.current
        if self.is_last() and self.is_first():
            card.remove()
            self._first = None
            self._last = None
            self._cur = None
            self.__size = 0
            return
        elif self.is_last():
            self._last = self.current.prev
        elif self.is_first():
            self._first = self.current.next
        card.remove()
        self.__size -= 1

    def is_last(self, card: TCard = None) -> bool:
        if card is not None:
            return card is self._last
        else:
            return self.current is self._last

    def is_first(self, card: TCard = None) -> bool:
        if card is not None:
            return card is self._first
        else:
            return self.current is self._first

    def isEmpty(self):
        return self._first is None

    @property
    def next(self) -> TCard:
        if self._cur is not None:
            self._cur = self._cur.next
        return self.current

    @property
    def prev(self) -> TCard:
        if self._cur is not None:
            self._cur = self._cur.prev
        return self.current
        
    def insert(self, card : TCard, pos = 0) -> bool:
        '''
        position:
            add first: 2
            add last: 1
            add after current: 0
        '''
        self.__size += 1
        if pos == 2:
            self.first = card
            return self.is_first(card)
        if pos == 1:
            self.last = card
            return self.is_last(card)
        if pos == 0:
            if self.isEmpty():
                self.first = card
                return self.current is card
            if self.is_last():
                self.last = card
            else:
                self._cur.next.prev = card
                self._cur.next = card
            return self.current is card


    def search(self, card: TCard) -> list[TCard]:
        pass
