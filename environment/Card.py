from utils.utils import get_token, get_optimized_token
from torchtext.data.utils import ngrams_iterator
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
from mtgtools.PCard import PCard
from collections import Counter
import json
import copy
import re

class AICard(PCard):
    """
    AICard Class for AI inforamtion construction
    """
    # TODO Create DAteset and Dataloader
    def __init__(self, response_dict: dict, amount : int = 1):
        super().__init__(response_dict)
        # contruct attribute list from incoming card attributes
        # maybe need test on special attributes are given
        self.__cxt_ids : list = None
        self.data_dict : dict = None
        self.data : MTGDataset = MTGDataset()
        self.input_layer_size : int = 0
        self.keys : list = [ "amount", "name", "mana_cost", "type_line" ]
        if getattr(self, "mana_cost", None) is not None:
            self.keys.append("mana_cost")
        if getattr(self, "power", None) is not None:
            self.keys.append("power")
        if getattr(self, "toughness", None) is not None:
            self.keys.append("toughness")
        if getattr(self, "loyalty", None) is not None:
            self.keys.append("loyalty")
        oracle_text = getattr(self, "oracle_text", None)
        if oracle_text is not None:
            self.oracle_text = re.sub( r'\(.*\)', "", self.oracle_text)
            self.keys.append("oracle_text")

        self.__wordlist : Counter = Counter()
        self.amount = amount
        
    @staticmethod
    def load_from_pcard(pcard: PCard):
        return AICard(json.loads(pcard.json.lower()))

    def create_dataset(self, word2idx: dict) -> dict:
        data_dict = dict()
        data_dict.update({"amount": [word2idx[" ".join(get_token(str(self.amount)))]]})
        data_dict.update({"name": [word2idx[" ".join(get_token(self.name))]]})
        if self.mana_cost is not None:
            data_dict.update({"mana_cost": [word2idx[self.mana_cost]]})
        data_dict.update({"type_line": [word2idx[t] for t in get_token(self.type_line)]})
        if self.power is not None:
            data_dict.update({"power": [word2idx[self.power]]})
        if self.toughness is not None:
            data_dict.update({"toughness": [word2idx[self.toughness]]})
        if self.loyalty is not None:
            data_dict.update({"loyalty": [word2idx[self.loyalty]]})
        if self.oracle_text is not None:
            data_dict.update({"oracle_text": [word2idx[t] for t in get_optimized_token(self.oracle_text, sorted(word2idx, key=word2idx.get, reverse=True))]})

        for k in data_dict.keys():
            self.input_layer_size += 1
            self.input_layer_size += len(data_dict[k])
        self.data_dict = data_dict

        data = dict()
        for k in self.data_dict.keys():
            data.update({k:  [ " ".join([ str(v) for v in  self.data_dict[k] ]) ] } )
        self.data = MTGDataset(pd.DataFrame.from_dict(data))
        return data


    def wordlist(self) -> Counter:
        if len(self.__wordlist):
            return self.__wordlist

        #print([ " ".join(get_token(self.name)) ])
        self.__wordlist.update(  { " ".join(get_token(self.name)): 9999 } )
        self.type_line = self.type_line.replace(" - "," ")
        for tltoken in get_token(self.type_line):
            self.__wordlist.update( { tltoken: 9999 } )

        self.__wordlist.update(Counter([ str(self.amount) ]))
        if self.mana_cost is not None:
            self.__wordlist.update({ self.mana_cost: 9999 })
        if self.power is not None:
            self.__wordlist.update(Counter([ self.power ]))
        if self.toughness is not None:
            self.__wordlist.update(Counter([ self.toughness ]))
        if self.loyalty is not None:
            self.__wordlist.update( Counter([ self.loyalty ]))
        if self.oracle_text is not None:
            self.__wordlist.update( Counter( list(ngrams_iterator(get_token(self.oracle_text),5))))
        for k in self.keys:
            self.__wordlist.update({ k: 9999 })
        return self.__wordlist


class MTGDataset(Dataset):
    def __init__(self, data: pd.DataFrame = pd.DataFrame(), input_size: int = 0, transform=None):
        """
        Arguments:
            data (pandas.DataFrame): Indexed card attributes, one row per card
            input_size (int): Size of the one-hot input vector per card
        """
        self.dataset: pd.DataFrame = data
        self.input_size: int = input_size

    def __str__(self):
        return str(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.zeros(self.input_size)
        pos = idx % max(1, self.input_size)
        x[pos] = 1.0
        return x, x

    def concat(self, data: pd.DataFrame):
        self.dataset = pd.concat([self.dataset, data])
        return self.dataset
