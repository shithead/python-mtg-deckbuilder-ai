from typing import Type, TypeVar

TCard = TypeVar('TCard', bound='Card')

class Card():
    def __init__(self):
        self._next: TCard = None
        self._prev: TCard = None

    @property
    def next(self) -> TCard:
        return self._next

    @next.setter
    def next(self, card: TCard):
        if card is not None:
            card._next = self._next
            card._prev = self
        self._next = card

    @property
    def prev(self) -> TCard:
        return self._prev

    @prev.setter
    def prev(self, card: TCard):
        if card is not None:
            card._prev = self._prev
            card._next = self
        self._prev = card

    def remove(self) -> TCard:
        self.next._prev = self.prev
        self.prev._next = self.next
        self.next = None
        self.prev = None
        return self

class MTGCard(Card):
    def __init__(self, attributes: dict):
        super().__init__()
        # contruct attribute list from incoming card attributes
        # maybe need test on special attributes are given TODO
        self.__keywords = [ "id", "name", "image_uris", "mana_cost",
                     "cmc", "type_line", "oracle_text", "power",
                     "toughness", "colors", "color_identity",
                     "keywords", "legalities", "set", "set_name",
                     "rarity", "edhrec_rank", "penny_rank"]
        print(attributes)
        for k, v in attributes.items():
            if k in self.__keywords:
                setattr(self, k, v)

    def __str__(self):
        ret_str = ""

        for keyw in self.__keywords:
            var = getattr(self, keyw, "None")
            ret_str += f"{keyw}: {var}\n"

        return ret_str
