import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder network that reconstructs observations from latent state (h, z).
    Supports vector and image observations.
    """
    def __init__(self, obs_type='vector', obs_shape=[4], hidden_dim=256, latent_dim=64):
        super().__init__()
        self.obs_type = obs_type
        self.obs_shape = obs_shape

        if obs_type == 'vector':
            obs_dim = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim),
            )
        elif obs_type == 'image':
            channels, height, width = obs_shape
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, 128 * (height // 8) * (width // 8)),
                nn.ReLU(),
                nn.Unflatten(1, (128, height // 8, width // 8)),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),  # up to h/4, w/4
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # up to h/2, w/2
                nn.ReLU(),
                nn.ConvTranspose2d(32, channels, 4, 2, 1),  # up to h, w
                nn.Sigmoid(),  # Assuming normalized images
            )
        else:
            raise ValueError("obs_type must be 'vector' or 'image'")

    def forward(self, h, z):
        """
        Args:
            h: (batch, hidden_dim)
            z: (batch, latent_dim)
        Returns:
            obs: (batch, *obs_shape)
        """
        input_ = torch.cat([h, z], dim=-1)
        return self.net(input_)