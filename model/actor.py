import torch
import torch.nn as nn
import torch.distributions as dist

class Actor(nn.Module):
    """
    Actor network that outputs action distribution from latent state.
    Supports discrete and continuous actions.
    """
    def __init__(self, action_dim=2, hidden_dim=256, latent_dim=64, is_continuous=False, quantize=False):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_dim = action_dim
        output_dim = 2 * action_dim if is_continuous else action_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.quantize = quantize
        if quantize:
            # Placeholder for quantization
            pass

    def forward(self, h, z):
        """
        Args:
            h: (batch, hidden_dim)
            z: (batch, latent_dim)
        Returns:
            logits: (batch, action_dim)
        """
        input_ = torch.cat([h, z], dim=-1)
        return self.net(input_)

    def get_action_dist(self, h, z):
        """
        Returns:
            dist.Categorical or dist.Normal
        """
        output = self.net(torch.cat([h, z], dim=-1))
        if self.is_continuous:
            mean, log_std = output.chunk(2, dim=-1)
            std = torch.exp(log_std)
            return dist.Normal(mean, std)
        else:
            return dist.Categorical(logits=output)

    def sample_action(self, h, z, deterministic=False, action_low=None, action_high=None, safety_threshold=0.8):
        """
        Sample action with safety checks.
        Args:
            h, z: as above
            deterministic: if True, take mean/argmax
            action_low, action_high: for clamping continuous actions
            safety_threshold: uncertainty threshold for safe actions
        Returns:
            action: (batch,) long tensor for discrete, (batch, action_dim) for continuous
        """
        dist_ = self.get_action_dist(h, z)
        uncertainty = self.get_uncertainty(h, z)

        # Safety: if uncertainty too high, take safe action (e.g., zero or mean)
        if self.is_continuous:
            safe_action = torch.zeros_like(dist_.mean)  # Safe: no torque
        else:
            safe_action = torch.tensor(0, dtype=torch.long)  # Safe discrete action

        if torch.any(uncertainty > safety_threshold):
            return safe_action

        if self.is_continuous:
            if deterministic:
                action = dist_.mean
            else:
                action = dist_.sample()
            if action_low is not None and action_high is not None:
                action = torch.clamp(action, action_low, action_high)
            return action
        else:
            if deterministic:
                return dist_.probs.argmax(dim=-1)
            else:
                return dist_.sample()

    def get_uncertainty(self, h, z):
        """
        Get action uncertainty (std for continuous, entropy for discrete).
        Useful for safety in real-world apps.
        """
        dist_ = self.get_action_dist(h, z)
        if self.is_continuous:
            return dist_.stddev.mean(dim=-1)  # Avg std across dims
        else:
            return dist_.entropy()  # Uncertainty in choice