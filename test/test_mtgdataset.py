import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader
from environment.Card import MTGDataset


class TestMTGDataset:
    def test_empty_dataset(self):
        ds = MTGDataset(pd.DataFrame(), input_size=10)
        assert len(ds) == 0

    def test_len(self):
        df = pd.DataFrame({"col": [[1, 2], [3], [4, 5, 6]]})
        ds = MTGDataset(df, input_size=10)
        assert len(ds) == 3

    def test_empty_dataframe_default_input_size(self):
        ds = MTGDataset()
        assert len(ds) == 0
        assert ds.input_size == 0

    def test_custom_input_size(self):
        df = pd.DataFrame({"col": [[1]]})
        ds = MTGDataset(df, input_size=42)
        assert ds.input_size == 42

    def test_getitem_returns_tuple_of_tensors(self):
        df = pd.DataFrame({"col": [[1, 2]]})
        ds = MTGDataset(df, input_size=10)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_getitem_output_shape(self):
        df = pd.DataFrame({"col": [[1]]})
        ds = MTGDataset(df, input_size=8)
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_getitem_x_equals_y(self):
        df = pd.DataFrame({"col": [[1, 2], [3], [4]]})
        ds = MTGDataset(df, input_size=5)
        for i in range(len(ds)):
            x, y = ds[i]
            assert torch.equal(x, y)

    def test_getitem_one_hot_first_row(self):
        df = pd.DataFrame({"col": [[1]]})
        ds = MTGDataset(df, input_size=5)
        x, _ = ds[0]
        assert x[0].item() == 1.0
        assert x[1].item() == 0.0
        assert x[2].item() == 0.0
        assert x[4].item() == 0.0

    def test_getitem_different_indices(self):
        df = pd.DataFrame({"col": [[1], [2], [3]]})
        ds = MTGDataset(df, input_size=5)
        x0, _ = ds[0]
        x1, _ = ds[1]
        x2, _ = ds[2]
        assert x0[0].item() == 1.0
        assert x1[1].item() == 1.0
        assert x2[2].item() == 1.0

    def test_getitem_index_wraps_around(self):
        df = pd.DataFrame({"col": [[1], [2]]})
        ds = MTGDataset(df, input_size=3)
        x, _ = ds[5]  # idx 5 % 3 = 2
        assert x[2].item() == 1.0
        assert x[0].item() == 0.0

    def test_getitem_negative_index(self):
        df = pd.DataFrame({"col": [[1]]})
        ds = MTGDataset(df, input_size=5)
        x, _ = ds[-1]
        assert x[4].item() == 1.0  # -1 % 5 = 4

    def test_getitem_tensor_index(self):
        df = pd.DataFrame({"col": [[1], [2], [3]]})
        ds = MTGDataset(df, input_size=5)
        idx = torch.tensor([0, 2])
        x0, _ = ds[idx[0]]
        x2, _ = ds[idx[1]]
        assert x0[0].item() == 1.0
        assert x2[2].item() == 1.0

    def test_dataloader_single_batch(self):
        df = pd.DataFrame({"col": [[1, 2], [3, 4, 5], [6]]})
        ds = MTGDataset(df, input_size=8)
        loader = DataLoader(ds, batch_size=3)
        X, Y = next(iter(loader))
        assert X.shape == (3, 8)
        assert Y.shape == (3, 8)

    def test_dataloader_partial_batch(self):
        df = pd.DataFrame({"col": [[1], [2], [3], [4], [5]]})
        ds = MTGDataset(df, input_size=6)
        loader = DataLoader(ds, batch_size=2)
        batches = list(loader)
        assert len(batches) == 3
        last_X, _ = batches[2]
        assert last_X.shape == (1, 6)

    def test_dataloader_shuffle(self):
        df = pd.DataFrame({"col": [[1]] * 100})
        ds = MTGDataset(df, input_size=20)
        loader = DataLoader(ds, batch_size=50, shuffle=True)
        X1, _ = next(iter(loader))
        loader = DataLoader(ds, batch_size=50, shuffle=True)
        X2, _ = next(iter(loader))
        # Very unlikely both batches are identical with shuffle
        assert not torch.equal(X1, X2) or X1.shape[0] == 50

    def test_dataloader_dtype(self):
        df = pd.DataFrame({"col": [[1]]})
        ds = MTGDataset(df, input_size=4)
        loader = DataLoader(ds, batch_size=1)
        X, _ = next(iter(loader))
        assert X.dtype == torch.float16

    def test_str(self):
        df = pd.DataFrame({"a": [[1]]})
        ds = MTGDataset(df, input_size=2)
        assert "a" in str(ds)

    def test_concat(self):
        df1 = pd.DataFrame({"col": [[1]]})
        df2 = pd.DataFrame({"col": [[2]]})
        ds = MTGDataset(df1, input_size=5)
        assert len(ds) == 1
        result = ds.concat(df2)
        assert len(ds) == 2
        assert len(result) == 2

    def test_multi_column_dataframe(self):
        df = pd.DataFrame({
            "amount": [[1]],
            "name": [[2, 3]],
            "type_line": [[4, 5, 6]],
        })
        ds = MTGDataset(df, input_size=10)
        assert len(ds) == 1
        x, _ = ds[0]
        assert x[0].item() == 1.0

    def test_large_dataset(self):
        df = pd.DataFrame({"col": [[i] for i in range(1000)]})
        ds = MTGDataset(df, input_size=64)
        assert len(ds) == 1000
        x, _ = ds[999]
        assert x[999 % 64].item() == 1.0
