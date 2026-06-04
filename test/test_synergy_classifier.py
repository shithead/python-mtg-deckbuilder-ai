import pytest
import torch
from ai.SynergyClassifier import SynergyClassifier


class TestSynergyClassifier:
    @pytest.fixture
    def model(self):
        return SynergyClassifier()

    def test_forward_output_shape(self, model):
        x = torch.randn(4, 768)
        out = model(x)
        assert out.shape == (4, 1)

    def test_forward_output_in_range(self, model):
        x = torch.randn(8, 768)
        out = model(x)
        assert torch.all(out >= 0) and torch.all(out <= 1)

    def test_forward_batch_independence(self, model):
        model.eval()
        x1 = torch.randn(2, 768)
        x2 = torch.cat([x1, torch.randn(2, 768)])
        out_full = model(x2)
        out_first = model(x1)
        assert torch.allclose(out_full[:2], out_first, atol=1e-6)

    def test_trainable_parameters(self, model):
        params = sum(p.numel() for p in model.parameters())
        assert 100_000 < params < 300_000

    def test_custom_dims(self):
        model = SynergyClassifier(card_embed_dim=128, hidden_dim=64)
        x = torch.randn(2, 256)
        out = model(x)
        assert out.shape == (2, 1)
