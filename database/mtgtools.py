import transaction

from mtgtools.MtgDB import MtgDB, get_scryfall_card_bulks
from mtgtools.PCardList import PCardList
from environment.Card import AICard
from config import DbPROVIDER, ZODB_PATH


class Database(MtgDB):
    def __init__(self, provider : int = DbPROVIDER["scryfall"], force_update : bool = False):
        super().__init__(ZODB_PATH)
        self.__provider = provider
        if force_update or self._needs_update():
            if self.__provider == DbPROVIDER["scryfall"]:
                self.scryfall_bulk_update()
            if self.__provider == DbPROVIDER["mtgio"]:
                self.mtgio_update()

    def _needs_update(self) -> bool:
        attr = (
            "scryfall_cards"
            if self.__provider == DbPROVIDER["scryfall"]
            else "mtgio_cards"
        )
        try:
            cards = getattr(self.root, attr, None)
            if cards is None or len(cards) == 0:
                return True
        except Exception:
            return True

        if self.__provider == DbPROVIDER["scryfall"]:
            try:
                bulk_data = get_scryfall_card_bulks()
                bulk_type = next(
                    (b for b in bulk_data["data"] if b["type"] == "default_cards"),
                    None,
                )
                if bulk_type:
                    latest = bulk_type["updated_at"]
                    stored = getattr(self.root, "_scryfall_updated_at", None)
                    if stored is None or latest > stored:
                        return True
            except Exception:
                return True

        return False

    def scryfall_bulk_update(self, bulk_type="default_cards", verbose=True):
        try:
            super().scryfall_bulk_update(bulk_type, verbose)
        except ValueError:
            # mtgtools bug: PSetList.where() uses substring matching, so
            # obsolete sets whose codes are substrings of other obsolete
            # codes (e.g. 'plist' in 'uplist') get added to obsolete_sets
            # multiple times, crashing on the second removal attempt.
            # Abort the partial transaction and retry with deduplicated sets.
            transaction.abort()
            if verbose:
                print("\nRetrying after deduplicating obsolete sets...")
            # Use exact code matching to safely remove obsolete sets
            from mtgtools.util.api_requests import get_response_json, scryfall_sets_url
            current_sets = self.root.scryfall_sets
            api_codes = {d["code"] for d in get_response_json(scryfall_sets_url)["data"]}
            removed = set()
            for pset in list(current_sets):
                if pset.code not in api_codes and pset.code not in removed:
                    current_sets.remove(pset)
                    removed.add(pset.code)
            transaction.commit()
            super().scryfall_bulk_update(bulk_type, verbose)

        bulk_data = get_scryfall_card_bulks()
        bulk_type_data = next(
            (b for b in bulk_data["data"] if b["type"] == bulk_type),
            None,
        )
        if bulk_type_data:
            self.root._scryfall_updated_at = bulk_type_data["updated_at"]
            transaction.commit()

    @property
    def cards(self) -> PCardList:
        if self.__provider == DbPROVIDER["scryfall"]:
            return self.root.scryfall_cards
        if self.__provider == DbPROVIDER["mtgio"]:
            return self.root.mtgio_cards
        return PCardList()

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

    def loadPool(self) -> PCardList:
        cards = self.root.basic_collection.filtered(lambda card: card.legalities['modern'] == 'legal' or card.legalities['modern'] == 'banned')
        if cards is None:
            raise RuntimeError("import first your basic collection via import_collection.py")
        return PCardList([ AICard.load_from_pcard(pcard) for pcard in cards ])

    def loadWccPool(self) -> PCardList:
        cards = self.root.wcc_collection
        if cards is None:
            raise RuntimeError("import first your wcc collection via import_collection.py")
        return PCardList([ AICard.load_from_pcard(pcard) for pcard in cards ])
