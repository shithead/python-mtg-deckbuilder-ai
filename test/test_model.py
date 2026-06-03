import pytest
import torch
from ai.MTGDeckBuilderModel import MTGDeckBuilderModel


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

    def test_forward_no_nan(self):
        model = MTGDeckBuilderModel(
            input_size=12,
            num_hidden_layer=3,
            output_size=7,
            device="cpu"
        )
        x = torch.randn(8, 12)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_forward_deterministic(self):
        model = MTGDeckBuilderModel(
            input_size=6,
            num_hidden_layer=2,
            output_size=4,
            device="cpu"
        )
        model.eval()
        x = torch.randn(3, 6)
        out1 = model(x)
        out2 = model(x)
        assert torch.equal(out1, out2)

    def test_forward_output_range_is_finite(self):
        model = MTGDeckBuilderModel(
            input_size=10,
            num_hidden_layer=5,
            output_size=3,
            device="cpu"
        )
        x = torch.randn(32, 10)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_forward_many_hidden_layers(self):
        model = MTGDeckBuilderModel(
            input_size=4,
            num_hidden_layer=20,
            output_size=2,
            device="cpu"
        )
        x = torch.randn(1, 4)
        out = model(x)
        assert out.shape == (1, 2)

    def test_forward_same_io_size(self):
        model = MTGDeckBuilderModel(
            input_size=32,
            num_hidden_layer=4,
            output_size=32,
            device="cpu"
        )
        x = torch.randn(10, 32)
        out = model(x)
        assert out.shape == (10, 32)
