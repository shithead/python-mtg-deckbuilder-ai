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
