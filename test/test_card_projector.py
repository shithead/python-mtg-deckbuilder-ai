import pytest
import torch
from ai.CardProjector import CardProjector


class TestCardProjector:
    @pytest.fixture
    def projector(self):
        return CardProjector()

    def test_forward_shape_1d(self, projector):
        x = torch.randn(384)
        out = projector(x)
        assert out.shape == (64,)

    def test_forward_shape_2d(self, projector):
        x = torch.randn(8, 384)
        out = projector(x)
        assert out.shape == (8, 64)

    def test_forward_output_is_finite(self, projector):
        x = torch.randn(4, 384)
        out = projector(x)
        assert torch.all(torch.isfinite(out))

    def test_forward_differentiable(self, projector):
        x = torch.randn(2, 384, requires_grad=True)
        out = projector(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 384)

    def test_forward_deterministic(self, projector):
        projector.eval()
        x = torch.randn(1, 384)
        e1 = projector(x)
        e2 = projector(x)
        assert torch.allclose(e1, e2)

    def test_trainable_parameters(self, projector):
        params = sum(p.numel() for p in projector.parameters())
        assert params == 384 * 128 + 128 + 128 * 64 + 64

    def test_custom_dims(self):
        proj = CardProjector(input_dim=128, hidden_dim=64, output_dim=32)
        x = torch.randn(3, 128)
        out = proj(x)
        assert out.shape == (3, 32)
