from utils.utils import get_token
from torchtext.data.utils import ngrams_iterator
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from mtgtools.PCard import PCard
import json
import re

class AICard(PCard):
    """
    AICard Class for AI inforamtion construction
    """
    # TODO Create DAteset and Dataloader
    def __init__(self, response_dict: dict):
        super().__init__(response_dict)
        # contruct attribute list from incoming card attributes
        # maybe need test on special attributes are given
        self.__cxt_ids : list = None
        self.data_dict : dict = None
        self.data : MTGDataset = None
        self.input_layer_size : int = 0
        self.keys : list = [ "name", "mana_cost", "type_line" ]
        
    @staticmethod
    def load_from_pcard(pcard: PCard):
        return AICard(json.loads(pcard.json))

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
        if self.oracle_text is not None:
            oracle_text = self.oracle_text
            for word in wordlist:
                oracle_text.replace(word,word2idx[word])

            data_dict.update({"oracle_text": get_token(oracle_text)})

        self.data_dict = data_dict
        self.data.concat(pd.DataFrame(data_dict))


    def wordlist(self):
        wordlist = list()

        wordlist.append(self.name)
        wordlist.append("name")
        wordlist.append(self.mana_cost)
        wordlist.append("mana_cost")
        wordlist.extend(get_token(self.type_line))
        wordlist.append("type_line")
        if self.power is not None:
            wordlist.append("power")
            wordlist.append(self.power)
            self.keys.append("power")
        if self.toughness is not None:
            wordlist.append("toughness")
            wordlist.append(self.toughness)
            self.keys.append("toughness")
        if self.loyalty is not None:
            wordlist.append("loyalty")
            wordlist.append(self.loyalty)
            self.keys.append("loyalty")
        if self.oracle_text is not None:
            oracle_text = self.oracle_text
            oracle_text = re.sub( r'\(.*\)', "", oracle_text)
            for rn in wordlist:
                oracle_text = oracle_text.replace(rn,"")
                self.input_layer_size += 1
            wordlist.append("oracle_text")
            self.keys.append("oracle_text")
            wordlist.extend(get_token(oracle_text))
        self.__wordlist = wordlist 
        self.input_layer_size += len(wordlist)

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
