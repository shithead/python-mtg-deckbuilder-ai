import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader
from ai.MTGDeckBuilderModel import MTGDeckBuilderModel
from environment.Card import MTGDataset


class TestMTGDataset:
    def test_len(self):
        df = pd.DataFrame({"col": [[1, 2], [3], [4, 5, 6]]})
        ds = MTGDataset(df, input_size=10)
        assert len(ds) == 3

    def test_getitem_output_type(self):
        df = pd.DataFrame({"col": [[1, 2]]})
        ds = MTGDataset(df, input_size=10)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (10,)
        assert y.shape == (10,)

    def test_getitem_one_hot(self):
        df = pd.DataFrame({"col": [[1]]})
        ds = MTGDataset(df, input_size=5)
        x, y = ds[0]
        assert x[0].item() == 1.0
        assert x[1].item() == 0.0
        assert torch.equal(x, y)

    def test_dataloader(self):
        df = pd.DataFrame({"col": [[1, 2], [3], [4, 5]]})
        ds = MTGDataset(df, input_size=8)
        loader = DataLoader(ds, batch_size=2)
        X, Y = next(iter(loader))
        assert X.shape == (2, 8)
        assert Y.shape == (2, 8)


class TestMTGDeckBuilderModel:
    def test_forward_output_shape(self):
        model = MTGDeckBuilderModel(
            input_size=10,
            num_hidden_layer=2,
            output_size=5,
            device="cpu"
        )
        x = torch.randn(1, 10)
        out = model(x)
        assert out.shape == (1, 5)

    def test_forward_with_different_input_size(self):
        model = MTGDeckBuilderModel(
            input_size=20,
            num_hidden_layer=1,
            output_size=3,
            device="cpu"
        )
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_forward_batch_size(self):
        model = MTGDeckBuilderModel(
            input_size=8,
            num_hidden_layer=2,
            output_size=2,
            device="cpu"
        )
        x = torch.randn(16, 8)
        out = model(x)
        assert out.shape == (16, 2)

    def test_forward_single_layer(self):
        model = MTGDeckBuilderModel(
            input_size=5,
            num_hidden_layer=0,
            output_size=5,
            device="cpu"
        )
        x = torch.randn(2, 5)
        out = model(x)
        assert out.shape == (2, 5)
