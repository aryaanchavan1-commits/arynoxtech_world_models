import torch
import torch.nn as nn

class RewardPredictor(nn.Module):
    """
    Reward predictor that predicts scalar reward from (h, z).
    """
    def __init__(self, hidden_dim=256, latent_dim=64, quantize=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.quantize = quantize

    def forward(self, h, z):
        """
        Args:
            h: (batch, hidden_dim)
            z: (batch, latent_dim)
        Returns:
            reward: (batch, 1)
        """
        input_ = torch.cat([h, z], dim=-1)
        return self.net(input_)