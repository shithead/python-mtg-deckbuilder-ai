import sys
import os

sys.path.append(os.path.abspath('../environment'))
sys.path.append(os.path.abspath('../database'))
sys.path.append(os.path.abspath('.'))
from database.mtgtools import Database
from environment.Card import AICard
from mtgtools.PCardList import PCardList
from environment.Deck import Deck
from utils.utils import get_token

from collections import Counter
import copy
import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from .MTGDeckBuilderModel import MTGDeckBuilderModel


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

torch.set_default_dtype(torch.float16)

class Preparer():
    """
    Preparer prepare Model with pool of cards. Every card is unique names.
    """
    def __init__(self, ModelClass: nn.Module, basic_data_path = "data"):

        self.__basic_path = basic_data_path
        self.pool : PCardList = Database().loadPool().unique_names()
        self.pool_size = len(self.pool)

        self.vocab : Counter = Counter()
        self.word2idx : dict = dict()
        self.embeddings = None
        self.input_layer_size : int  = 0
        self.keys :set = set()
        self.datasets : pd.DataFrame = pd.DataFrame()
        self.model : MTGDeckBuilderModel = None


        self.load()
        self.create_word2idx()
        self.create_datasets()
        self.embedding()

        ## clean space XXX for testing 
        self.pool = PCardList()
        self.vocab = Counter()

        if self.model is None:
            self.model = ModelClass(self.pool_size, 40, 4 * self.pool_size, device)

        summary(self.model)

    @property
    def Model(self):
        return self.model

    def load(self):
        for i, card in  enumerate(self.pool):
            print(f"\rload keys from card: {i+1}/{len(self.pool)}", end="", flush=True)
            self.keys.update(card.keys)
        
        #try:
        #    self.vocab = torch.load(f"{self.__basic_path}/vocable.bin")
        #except FileNotFoundError as e:
        #    print(e)
        #    pass
        try:
            self.word2idx = torch.load(f"{self.__basic_path}/word2idx.bin")
        except FileNotFoundError as e:
            print(e)
            pass
        try:
            self.embeddings = torch.load(f"{self.__basic_path}/embeddings.bin")
        except FileNotFoundError as e:
            print(e)
            pass
        try:
            self.datasets = pd.DataFrame.from_dict(torch.load(f"{self.__basic_path}/datasets.bin"))
        except FileNotFoundError as e:
            print(e)
            pass
        try:
            self.model = torch.load(f"{self.__basic_path}/t1_model_{len(self.pool)}_{len(self.keys)}.bin")
        except FileNotFoundError as e:
            print(e)
            pass

    def save(self):
        #torch.save(self.vocab, f"{self.__basic_path}/vocable.bin")
        torch.save(self.word2idx, f"{self.__basic_path}/word2idx.bin")
        torch.save(self.embeddings, f"{self.__basic_path}/embeddings.bin")
        torch.save(self.datasets.to_dict(), f"{self.__basic_path}/datasets.bin")
        torch.save(self.model.state_dict(), f"{self.__basic_path}/t1_model_{self.pool_size}_{len(self.keys)}.bin")


    def rate_vocab(self, vocab) -> list:
        vocabl = sorted(vocab, key=vocab.get)
        rated_vocab = dict()
        for i,v in enumerate(vocabl):
            print(f"\rrerate vocab: {i+1}/{len(vocabl)}", end="", flush=True)
            rate = vocab[v]
            tokens = get_token(v)
            if len(tokens) > 1:
                for t in tokens:
                    rate += vocab[t]
                rate = (vocabl.index(v)/len(vocabl)) * rate
            else:
                rate = 1
            rated_vocab.update({ v: int(rate) })
        return sorted(rated_vocab, key=rated_vocab.get, reverse=True)


    def embedding(self):
        if self.embeddings is not None:
            return self.embeddings

        self.embeddings = nn.Embedding(self.input_layer_size, self.pool_size)

    def create_word2idx(self):
        if len(self.word2idx) != 0:
            return

        # create vocab
        for i, card in  enumerate(self.pool):
            print(f"\rcreate vocab from card: {i+1}/{len(self.pool)}", end="", flush=True)
            self.vocab.update(card.wordlist())
            self.keys.update(card.keys)

        self.vocab = self.rate_vocab(self.vocab)
        vocab_size = len(self.vocab)

        # map words to unique indices
        for ind, word in  enumerate(self.vocab):
            print(f"\rcreate word2idx: {ind+1}/{len(self.vocab)}", end="", flush=True)
            self.word2idx.update({word: ind})

    def create_datasets(self):
        if self.datasets.size != 0:
            return

        import concurrent.futures

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix='AICard-dataset')
        
        global datasets_array
        datasets_array = [None] * len(self.pool)

        datasets = dict()
        for k in self.keys:
            datasets.update({k : []})

        def get_dataset(self, idx: int, card: AICard):
            global datasets_array
            datasets_array[idx] = card.create_dataset(self.word2idx, self.vocab)
            self.input_layer_size += card.input_layer_size


        for i, card in  enumerate(self.pool):
            print(f"\rcreate_dataset: {i+1}/{len(self.pool)}", end="", flush=True)
            pool.submit(get_dataset, self, i, card)
        
            
        print(f"\ncreate_dataset wait on Threads")
        pool.shutdown(wait =  True)
        print(f"\ncreate_dataset all Threads finished")

        for d in datasets_array:
            for k in self.keys:
                try:
                    datasets[k].append(d[k])
                except KeyError:
                    datasets[k].append([])

        self.datasets = pd.DataFrame().from_dict(datasets)

