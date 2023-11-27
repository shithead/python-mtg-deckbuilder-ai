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

torch.set_default_dtype(torch.float16)

class MTGDeckBuilderModel(nn.Module):
    def __init__(self, input_size, num_hidden_layer, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        diff = int((input_size - output_size) / num_hidden_layer)
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
    def __init__(self, ModelClass: nn.Module):
        self.pool : PCardList = Database().loadPool().unique_names()

        self.vocab : Counter = Counter()
        self.word2idx : dict = None
        self.embeddings = None
        self.input_layer_size : int  = 0
        self.keys :set = set()
        self.embedding()

        print(len(self.pool))
        # outpu_layer coded as binary code
        print(self.input_layer_size)
        ## XXX for testing
        self.model = ModelClass(self.input_layer_size, int(len(self.keys)/2), len(self.pool))
        # TODO maybe fit to card attributes
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

    @property
    def Model(self):
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), "data/model.bin")

    def rate_vocab(self, vocab) -> list:
        vocabl = sorted(vocab, key=vocab.get)
        rated_vocab = dict()
        for v in vocabl:
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
        # create vocab
        for i, card in  enumerate(self.pool):
            print(f"create vocab: {i+1}/{len(self.pool)}")
            self.vocab.update(card.wordlist())
            self.keys.update(card.keys)

        self.vocab = self.rate_vocab(self.vocab)
        torch.save(self.vocab, "data/vocable.bin")
        vocab_size = len(self.vocab)

        # map words to unique indices
        self.word2idx = {word: ind for ind, word in enumerate(self.vocab)} 
        torch.save(self.word2idx, "data/word2idx.bin")

        self.embeddings = nn.Embedding(vocab_size, int(len(self.keys)/2))
        torch.save(self.embeddings, "data/embeddings.bin")

        for i, card in  enumerate(self.pool):
            print(f"create_dataset: {i+1}/{len(self.pool)}")
            card.create_dataset(self.word2idx, self.vocab)
            self.input_layer_size += card.input_layer_size


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
