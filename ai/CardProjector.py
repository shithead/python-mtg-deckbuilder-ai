import torch
import torch.nn as nn


class CardProjector(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)
