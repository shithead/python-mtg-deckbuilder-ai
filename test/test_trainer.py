import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader

from ai.MTGDeckBuilderModel import MTGDeckBuilderModel
from environment.Card import MTGDataset


class TestTrainerDataloader:
    def test_create_dataloader_batch_size(self):
        pytest.importorskip("mtgtools")
        import ai.Trainer as trainer_module

        pool_size = 10
        df = pd.DataFrame({"col": [[i] for i in range(pool_size)]})

        trainer = object.__new__(trainer_module.Trainer_T1)
        trainer.pool_size = pool_size
        trainer.datasets = df

        loader = trainer.create_dataloader(batch_size=4, shuffle=False)
        X, Y = next(iter(loader))
        assert X.shape == (4, pool_size)
        assert Y.shape == (4, pool_size)

    def test_create_dataloader_default_params(self):
        pytest.importorskip("mtgtools")
        import ai.Trainer as trainer_module

        df = pd.DataFrame({"col": [[1], [2]]})
        trainer = object.__new__(trainer_module.Trainer_T1)
        trainer.pool_size = 8
        trainer.datasets = df

        loader = trainer.create_dataloader()
        X, Y = next(iter(loader))
        assert X.shape == (2, 8)

    def test_create_dataloader_shuffle_false(self):
        pytest.importorskip("mtgtools")
        import ai.Trainer as trainer_module

        df = pd.DataFrame({"col": [[i] for i in range(50)]})
        trainer = object.__new__(trainer_module.Trainer_T1)
        trainer.pool_size = 10
        trainer.datasets = df

        loader = trainer.create_dataloader(batch_size=50, shuffle=False)
        X1, _ = next(iter(loader))
        X2, _ = next(iter(DataLoader(
            MTGDataset(df, input_size=10), batch_size=50, shuffle=False
        )))
        assert torch.equal(X1, X2)


class TestTrainerTrainLoop:
    def test_train_loop_runs_without_error(self):
        pytest.importorskip("mtgtools")
        import ai.Trainer as trainer_module

        model = MTGDeckBuilderModel(input_size=4, num_hidden_layer=0, output_size=4)
        ds = MTGDataset(pd.DataFrame({"col": [[i] for i in range(8)]}), input_size=4)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)

        trainer = object.__new__(trainer_module.Trainer_T1)
        trainer.model = model

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer.train_loop(loader, loss_fn, optimizer)

    def test_test_loop_runs_without_error(self):
        pytest.importorskip("mtgtools")
        import ai.Trainer as trainer_module

        model = MTGDeckBuilderModel(input_size=4, num_hidden_layer=0, output_size=4)
        ds = MTGDataset(pd.DataFrame({"col": [[i] for i in range(8)]}), input_size=4)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)

        trainer = object.__new__(trainer_module.Trainer_T1)
        trainer.model = model

        loss_fn = torch.nn.MSELoss()

        trainer.test_loop(loader, loss_fn)

    def test_train_loop_reduces_loss(self):
        pytest.importorskip("mtgtools")
        import ai.Trainer as trainer_module

        model = MTGDeckBuilderModel(input_size=4, num_hidden_layer=0, output_size=4)
        ds = MTGDataset(pd.DataFrame({"col": [[i] for i in range(100)]}), input_size=4)
        loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)

        trainer = object.__new__(trainer_module.Trainer_T1)
        trainer.model = model

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def avg_loss(dataloader):
            losses = []
            for X, y in dataloader:
                pred = model(X)
                losses.append(loss_fn(pred, y).item())
            return sum(losses) / len(losses)

        loss_before = avg_loss(loader)

        for _ in range(5):
            for X, y in loader:
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        loss_after = avg_loss(loader)
        assert loss_after < loss_before
