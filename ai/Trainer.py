import sys
import os

sys.path.append(os.path.abspath('../environment'))
sys.path.append(os.path.abspath('../database'))
sys.path.append(os.path.abspath('.'))
from database.SQLite import SQLite
from environment.Card import MTGCard
from environment.Pool import Pool
from environment.Deck import Deck

from collections import Counter, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class MTGDeckBuilderModel(nn.Module):
    def __init__(self, input_size, num_hidden_layer, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        diff = int((input_size - output_size) / num_hidden_layer)
        l = input_size - diff
        layers = [
                ("input_layer", nn.Linear(input_size, l)),
                ("ReLU1", nn.ReLU())
                ]
        for k in range(num_hidden_layer):
            layers.extend(
                    [(f"hidden{k}",nn.Linear(l,l-diff)),
                    (f"ReLU{k+1}",nn.ReLU())]
                    )
            l -= diff
        layers.extend([("output_layer",nn.Linear(l,output_size))])
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
    def __init__(self, ModelClass: nn.Module):
        self.pool = SQLite(os.path.abspath("../MTG-search/mtg.db")).loadPool()
        def __unifyPool(self):
            card = self.pool.first
            last_id = card.id
            while (not self.pool.is_last()):
                card = self.pool.next
                if last_id == card.id:
                   del self.pool.current
                else:
                    last_id = card.id

        __unifyPool(self)
        self.keys = set(self.pool.first.keys)
        self.vocab = Counter(self.keys)
        self.word2idx = None
        self.embeddings = None
        self.embedding()
        self.model = ModelClass(len(self.keys)*len(self.pool), int(len(self.keys)/2), len(self.pool))
        # TODO maybe fit to card attributes
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

    @property
    def Model(self):
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), "model.pth")

    def embedding(self):
        # create vocab
        card : MTGCard = self.pool.first
        while (not self.pool.is_last()):
            self.vocab.update(Counter(card.wordlist()))
            card = self.pool.next
            self.keys.update(card.keys)
            self.vocab.update(Counter(self.keys))
        self.vocab.update(Counter(card.wordlist()))

        self.vocab = sorted(self.vocab, key=self.vocab.get, reverse=True)
        vocab_size = len(self.vocab)

        # map words to unique indices
        self.word2idx = {word: ind for ind, word in enumerate(self.vocab)} 

        self.embeddings = nn.Embedding(vocab_size, len(self.keys))

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
