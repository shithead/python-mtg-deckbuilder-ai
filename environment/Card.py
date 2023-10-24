from typing import Type, TypeVar
from utils.utils import fitstring
from torch.utils.data import Dataset, DataLoader

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
    """
    MTGCard Class for input attributes data
    """
    # TODO Create DAteset and Dataloader
    def __init__(self, attributes: dict):
        super().__init__()
        # contruct attribute list from incoming card attributes
        # maybe need test on special attributes are given
        self.__keywords = [ "id", "name", "image_uris", "mana_cost",
                     "cmc", "type_line", "oracle_text", "power",
                     "toughness", "colors", "color_identity",
                     "keywords", "legalities", "set", "set_name",
                     "rarity", "edhrec_rank", "penny_rank"]
        for k, v in attributes.items():
            if k in self.keys:
                setattr(self, k, v)

        # image_uris only from interesset by UI
        self.__keywords.remove("image_uris")
        self.__cxt_ids : list = None
        self.__data : MTGDataset = None

    def wordlist(self):
        wordlist = list()

        for k in self.keys:
            if not hasattr(self,k):
                continue
            if isinstance(getattr(self,k),dict):
                for attk, v in getattr(self,k).items():
                    wordlist.append(attk)
                    wordlist.extend(fitstring(v).split(" "))
            elif isinstance(getattr(self,k),list):
                for e in getattr(self,k):
                    wordlist.extend(fitstring(e).split(" "))
            elif isinstance(getattr(self,k),str):
                wordlist.extend(fitstring(getattr(self,k)).split(" "))
            else:
                wordlist.append(getattr(self,k))
        return wordlist

    @property
    def cxt_ids(self):
        return self.__cxt_ids
    
    def cxt_ids(self, word2idx: dict):
        if self.__cxt_ids is None:
            self.__cxt_ids = torch.tensor([word2idx[w] for w in self.wordlist()], dtype=torch.long)
            self.__data = MTGDataset(self.__cxt_ids)
        return self.__cxt_ids

    @property
    def keys(self):
        return self.__keywords

    def __str__(self):
        ret_str = ""

        for keyw in self.__keywords:
            var = getattr(self, keyw, "None")
            ret_str += f"{keyw}: {var}\n"

        return ret_str

class MTGDataset(Dataset):
    def __init__(self, data: list, transform=None):
        """
        Arguments:
            data (list): Indexed card attributes
        """
        self.dataset: list = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datas = self.dataset.iloc[idx, 0:]
        datas = np.array([datas], dtype=float).reshape(-1, 2)
        sample = {'datas': datas}

        return sample
