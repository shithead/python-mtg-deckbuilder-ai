from mtgtools.MtgDB import MtgDB
from mtgtools.PCardList import PCardList
from environment.Card import MTGCard
from environment.Pool import Pool


DbPROVIDER = dict({"scryfall" : 1, "mtgio": 2})


class Database(MtgDB):
    def __init__(self, provider : int = DbPROVIDER["scryfall"]):
        super().__init__('data/mtgdb.fs')
        self.__provider = provider
        if self.__provider == DbPROVIDER["scryfall"]:
            self.scryfall_bulk_update()
        if self.__provider == DbPROVIDER["mtgio"]:
            self.mtgio_update()

    def load_from_file(self,path: str) -> PCardList:
        """
        Comment lines can be specified with '//', possible desired sets
        can be specified with either '(set_code)' or '[set_code]' and
        sideboard cards with the prefix 'SB:'. The set brackets can be
        anywhere but the desired number of cards must come before the
        name of the card. If no matching set is found, a card from a random set is returned.
        """
        if self.__provider == DbPROVIDER["scryfall"]:
            return self.root.scryfall_cards.from_file(path)
        if self.__provider == DbPROVIDER["mtgio"]:
            return self.root.mtgio_cards.from_file(path)
        return PCardList()

    def load_from_str(self, string: str) -> PCardList:
        """
        Comment lines can be specified with '//', possible desired sets
        can be specified with either '(set_code)' or '[set_code]' and
        sideboard cards with the prefix 'SB:'. The set brackets can be
        anywhere but the desired number of cards must come before the
        name of the card. If no matching set is found, a card from a random set is returned.
        """
        if self.__provider == DbPROVIDER["scryfall"]:
            return self.root.scryfall_cards.from_str(string)
        if self.__provider == DbPROVIDER["mtgio"]:
            return self.root.mtgio_cards.from_str(string)
        return PCardList()

    def loadPool(self) -> Pool:
        pool = Pool()
        cards = self.root.basic_collection
        if cards is None:
            return Exception("import first your basic collection via import_collection.py")
        return pool

