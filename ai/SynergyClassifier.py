import torch
import torch.nn as nn


class SynergyClassifier(nn.Module):
    def __init__(self, card_embed_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        input_dim = card_embed_dim * 2
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)
