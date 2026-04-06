import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder network that maps observations to latent embeddings, with robustness to missing data.
    Supports vector and image observations.
    """
    def __init__(self, obs_type='vector', obs_shape=[4], latent_dim=64, hidden_dim=256):
        super().__init__()
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim

        if obs_type == 'vector':
            obs_dim = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            # Learnable embedding for missing values
            self.missing_embed = nn.Parameter(torch.randn(obs_dim))
        elif obs_type == 'image':
            # Simple CNN for images
            channels, height, width = obs_shape
            self.net = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2),  # (32, h/2, w/2)
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),  # (64, h/4, w/4)
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2),  # (128, h/8, w/8)
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # (128, 1, 1)
                nn.Flatten(),
                nn.Linear(128, latent_dim),
            )
            # For images, missing data not directly handled, assume full images or use masking in preprocessing
        else:
            raise ValueError("obs_type must be 'vector' or 'image'")

    def forward(self, obs, mask=None):
        """
        Args:
            obs: Tensor of shape (batch, *obs_shape)
            mask: Optional tensor for vector obs, shape (batch, obs_dim) indicating valid (1) or missing (0)
        Returns:
            Tensor of shape (batch, latent_dim)
        """
        if self.obs_type == 'vector':
            if mask is not None:
                # Replace missing values with learnable embedding
                missing_obs = self.missing_embed.expand_as(obs)
                obs = torch.where(mask, obs, missing_obs)
            return self.net(obs)
        elif self.obs_type == 'image':
            # For images, mask not used here, assume preprocessing handles missing
            return self.net(obs)