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

class Trainer_T1():
    """
    Trainer_T1 train Model with with pool of cards. Every card is unique
    """
    def __init__(self, ModelClass: nn.Module, basic_data_path = "data"):

        self.__basic_path = basic_data_path
        self.pool : PCardList = Database().loadPool().unique_names()
        self.pool_size = len(self.pool)

        self.word2idx : dict = dict()
        self.embeddings = None
        self.keys :set = set()
        self.datasets : pd.DataFrame = pd.DataFrame()
        self.model : MTGDeckBuilderModel = None


        self.load()
        summary(self.model)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= 0.001  )

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
