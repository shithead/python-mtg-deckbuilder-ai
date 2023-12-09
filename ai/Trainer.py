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

from collections import Counter, OrderedDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd


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

class MTGDeckBuilderModel(nn.Module):
    def __init__(self, input_size, num_hidden_layer, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        diff = int((input_size - output_size) / (num_hidden_layer+1))
        l = input_size - diff
        layers = [
                ("input_layer", nn.Linear(input_size, max(1,l), device=device)),
                ("ReLU1", nn.ReLU())
                ]
        for k in range(num_hidden_layer):
            if (l - diff) < output_size:
                break
            layers.extend(
                    [(f"hidden{k}",nn.Linear(max(1,l),max(1,l-diff), device=device)),
                    (f"ReLU{k+1}",nn.ReLU())]
                    )
            l -= diff
        layers.extend([("output_layer",nn.Linear(max(1,l),output_size, device=device))])
        self.linear_relu_stack = nn.Sequential(
                OrderedDict(layers)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Trainer_T1():
    """
    Trainer_T1 train Model with with pool of cards. Every card is unique
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


        self.load()
        self.embedding()
        self.create_datasets()

        print(self.pool_size)
        # outpu_layer coded as binary code
        print(self.input_layer_size)
        ## XXX for testing
        self.pool = PCardList()
        self.vocab = Counter()
        self.model = ModelClass(self.input_layer_size, max(2,len(self.keys)), self.pool_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

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

        self.embeddings = nn.Embedding(vocab_size, int(len(self.keys)/2))

    def create_datasets(self):
        if self.datasets.size != 0:
            return

        datasets = dict()
        for k in self.keys:
            datasets.update({k : []})

        for i, card in  enumerate(self.pool):
            print(f"\rcreate_dataset: {i+1}/{len(self.pool)}", end="", flush=True)

            d = card.create_dataset(self.word2idx, self.vocab)
            for k in self.keys:
                try:
                    datasets[k].append(d[k])
                except KeyError:
                    datasets[k].append([])


            self.input_layer_size += card.input_layer_size
        self.datasets = pd.DataFrame().from_dict(datasets)


    def train_loop(dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
    
        # Evaluating the self.model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
