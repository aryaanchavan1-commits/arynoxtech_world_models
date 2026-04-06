import torch
import torch.nn as nn
import torch.distributions as dist

class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM) with GRU for deterministic state.
    Optimized for efficiency on edge devices.
    """
    def __init__(self, action_dim=1, hidden_dim=256, latent_dim=64, is_continuous=False, quantize=False):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.is_continuous = is_continuous
        self.quantize = quantize

        input_dim = hidden_dim + (action_dim if is_continuous else action_dim) + latent_dim
        # GRU for deterministic state h
        self.gru = nn.GRUCell(input_dim, hidden_dim)

        # Prior network: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim),  # mean and log_std
        )

        # Posterior network: q(z_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 128),  # h_t and encoded_obs
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim),
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use observe_step or imagine_step")

    def observe_step(self, action, obs_embed, prev_h, prev_z):
        """
        Observe step: update state with observation.
        Args:
            action: (batch,) long tensor
            obs_embed: (batch, latent_dim) encoded observation
            prev_h: (batch, hidden_dim)
            prev_z: (batch, latent_dim)
        Returns:
            next_h: (batch, hidden_dim)
            next_z: (batch, latent_dim) sampled from posterior
            next_z_mean: (batch, latent_dim)
            next_z_std: (batch, latent_dim)
        """
        if self.is_continuous:
            action_input = action
        else:
            action_input = torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()

        # GRU input: [prev_h, action_input, prev_z]
        gru_input = torch.cat([prev_h, action_input, prev_z], dim=-1)
        next_h = self.gru(gru_input, prev_h)

        # Posterior: q(z | h, e)
        post_input = torch.cat([next_h, obs_embed], dim=-1)
        post_params = self.posterior_net(post_input)
        next_z_mean, next_z_log_std = post_params.chunk(2, dim=-1)
        next_z_std = torch.exp(next_z_log_std)

        # Sample with reparam trick
        eps = torch.randn_like(next_z_mean)
        next_z = next_z_mean + next_z_std * eps

        return next_h, next_z, next_z_mean, next_z_std

    def imagine_step(self, action, prev_h, prev_z):
        """
        Imagine step: predict next state without observation.
        Args:
            action: (batch,) long tensor
            prev_h: (batch, hidden_dim)
            prev_z: (batch, latent_dim)
        Returns:
            next_h: (batch, hidden_dim)
            next_z: (batch, latent_dim) sampled from prior
            next_z_mean: (batch, latent_dim)
            next_z_std: (batch, latent_dim)
        """
        if self.is_continuous:
            action_input = action
        else:
            action_input = torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()

        # GRU input: [prev_h, action_input, prev_z]
        gru_input = torch.cat([prev_h, action_input, prev_z], dim=-1)
        next_h = self.gru(gru_input, prev_h)

        # Prior: p(z | h)
        prior_params = self.prior_net(next_h)
        next_z_mean, next_z_log_std = prior_params.chunk(2, dim=-1)
        next_z_std = torch.exp(next_z_log_std)

        # Sample
        eps = torch.randn_like(next_z_mean)
        next_z = next_z_mean + next_z_std * eps

        return next_h, next_z, next_z_mean, next_z_std

    def prior_dist(self, h):
        """
        Get prior distribution for z given h.
        Args:
            h: (batch, hidden_dim)
        Returns:
            dist.Normal
        """
        params = self.prior_net(h)
        mean, log_std = params.chunk(2, dim=-1)
        std = torch.exp(log_std)
        return dist.Normal(mean, std)

    def posterior_dist(self, h, obs_embed):
        """
        Get posterior distribution for z given h and obs_embed.
        Args:
            h: (batch, hidden_dim)
            obs_embed: (batch, latent_dim)
        Returns:
            dist.Normal
        """
        input_ = torch.cat([h, obs_embed], dim=-1)
        params = self.posterior_net(input_)
        mean, log_std = params.chunk(2, dim=-1)
        std = torch.exp(log_std)
        return dist.Normal(mean, std)