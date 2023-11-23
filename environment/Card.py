from typing import Type, TypeVar
from utils.utils import get_token
from torchtext.data.utils import ngrams_iterator
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from mtgtools.PCard import PCard

TCard = TypeVar('TCard', bound='Card')

class Card(PCard):
    def __init__(self, response_dict:dict):
        super().__init__(response_dict)
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
    MTGCard Class for AI inforamtion construction
    """
    # TODO Create DAteset and Dataloader
    def __init__(self, response_dict: dict):
        super().__init__(response_dict)
        # contruct attribute list from incoming card attributes
        # maybe need test on special attributes are given
        self.__cxt_ids : list = None
        self.data : MTGDataset = None
        
    def create_dataset(self, word2idx: dict):
        data_dict = dict()
        data_dict.update({"name": [word2idx[self.name]]})
        data_dict.update({"mana_cost": [word2idx[self.mana_cost]]})
        data_dict.update({"type_line": [word2idx[t] for t in get_token(self.type_line)]})
        if self.power is not None:
            data_dict.update({"power": [word2idx[self.power]]})
        if self.thoughness is not None:
            data_dict.update({"toughness": [word2idx[self.toughness]]})
        if self.loyality is not None:
            data_dict.update({"loyality": [word2idx[self.loyality]]})
        oracle_text = self.oracle_text
        for word in wordlist:
            oracle_text.replace(word,word2idx[word])

        data_dict.update({"oracle_text": get_token(oracle_text)})


        self.data.concat(pd.DataFrame(data_dict))


    def wordlist(self):
        wordlist = list()

        wordlist.append(self.name)
        wordlist.append(self.mana_cost)
        wordlist.extend(get_token(self.type_line))
        if self.power is not None:
            wordlist.append(self.power)
        if self.thoughness is not None:
            wordlist.append(self.toughness)
        if self.loyality is not None:
            wordlist.append(self.loyality)
        oracle_text = self.oracle_text
        for rn in wordlist:
            oracle_text = oracle_text.replace(rn,"")
        wordlist.extend(get_token(oracle_text))
        self.__wordlist = wordlist 

        return wordlist

    @property
    def cxt_ids(self):
        return self.__cxt_ids
    
    def cxt_ids(self, word2idx: dict):
        if self.__cxt_ids is None:
            self.__cxt_ids = torch.tensor([word2idx[w] for w in self.wordlist()], dtype=torch.long)
        return self.__cxt_ids


class MTGDataset(Dataset):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), transform=None):
        """
        Arguments:
            data (pandas.DataFrame): Indexed card attributes
        """
        self.dataset: pd.DataFrame = data

    def __str__(self):
        return str(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datas = self.dataset.iloc[idx, 0:]
        datas = np.array([datas], dtype=float).reshape(-1, 2)
        sample = {'datas': datas}

        return sample

    def concat(self, data : pd.DataFrame):
        self.dataset = pd.concat([self.dataset, data])
        return self.dataset
