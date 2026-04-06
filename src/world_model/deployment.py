"""
Deployment module for industrial and real-world use of the World Model.
Supports inference on edge devices, robots, drones, etc.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from model.encoder import Encoder
from model.rssm import RSSM
from model.actor import Actor
from model.critic import Critic
from model.reward_predictor import RewardPredictor

class WorldModelAgent:
    """
    Deployable agent for real-world tasks.
    Handles observation preprocessing, action selection, and adaptation to messy environments.
    """
    def __init__(self, config_path='config.json', model_path='models/'):
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.config = config
        self.obs_type = config.get('obs_type', 'vector')
        self.obs_shape = config.get('obs_shape', [4])
        self.action_type = config.get('action_type', 'discrete')
        self.is_continuous = self.action_type == 'continuous'
        self.action_dim = config.get('action_dim', 2)
        self.action_low = config.get('action_low', None)
        self.action_high = config.get('action_high', None)
        self.device = torch.device(config.get('device', 'cpu'))

        # Load models
        self.encoder = Encoder(obs_type=self.obs_type, obs_shape=self.obs_shape, latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim'])
        self.rssm = RSSM(action_dim=self.action_dim, hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'], is_continuous=self.is_continuous)
        self.actor = Actor(action_dim=self.action_dim, hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'], is_continuous=self.is_continuous)
        self.reward_pred = RewardPredictor(hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'])

        self.load_models(model_path)
        self.to_device()

        # State
        self.h = torch.zeros(1, self.rssm.hidden_dim).to(self.device)
        self.z = self.rssm.prior_dist(self.h).sample()

        # Logging
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_models(self, model_path):
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'encoder.pth'), map_location=self.device))
        self.rssm.load_state_dict(torch.load(os.path.join(model_path, 'rssm.pth'), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(model_path, 'actor.pth'), map_location=self.device))
        self.reward_pred.load_state_dict(torch.load(os.path.join(model_path, 'reward_pred.pth'), map_location=self.device))
        self.encoder.eval()
        self.rssm.eval()
        self.actor.eval()
        self.reward_pred.eval()

    def to_device(self):
        self.encoder.to(self.device)
        self.rssm.to(self.device)
        self.actor.to(self.device)
        self.reward_pred.to(self.device)

    def preprocess_obs(self, obs, mask=None):
        """
        Preprocess observation: add noise, handle missing data.
        """
        obs = np.array(obs, dtype=np.float32)
        if self.obs_type == 'vector':
            # Add noise
            obs += np.random.normal(0, self.config.get('obs_noise_std', 0.01), size=obs.shape)
            if mask is not None:
                obs = np.where(mask, obs, 0.0)  # Set missing to 0
        # For image, assume already processed
        return obs

    def reset(self):
        self.h = torch.zeros(1, self.rssm.hidden_dim).to(self.device)
        self.z = self.rssm.prior_dist(self.h).sample()

    def step(self, obs, mask=None):
        """
        Take a step: process obs, update state, select action.
        """
        obs_processed = self.preprocess_obs(obs, mask)
        obs_tensor = torch.tensor(obs_processed, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Encode
        obs_embed = self.encoder(obs_tensor, mask)

        # Dummy action for observe
        dummy_action = torch.zeros(1, self.action_dim if self.is_continuous else 1, dtype=torch.float32 if self.is_continuous else torch.long).to(self.device)
        self.h, self.z, _, _ = self.rssm.observe_step(dummy_action, obs_embed, self.h, self.z)

        # Sample action with safety
        safety_threshold = self.config.get('safety_threshold', 0.8)
        action_tensor = self.actor.sample_action(self.h, self.z, deterministic=True, action_low=self.action_low, action_high=self.action_high, safety_threshold=safety_threshold)
        if self.is_continuous:
            action = action_tensor.squeeze(0).cpu().numpy()
        else:
            action = action_tensor.item()

        return action

    def imagine_trajectory(self, horizon=10):
        """
        Imagine future trajectory for planning.
        """
        h = self.h.clone()
        z = self.z.clone()
        actions = []
        rewards = []
        uncertainties = []
        for _ in range(horizon):
            action = self.actor.sample_action(h, z, deterministic=False)
            uncertainty = self.actor.get_uncertainty(h, z).item()
            actions.append(action)
            uncertainties.append(uncertainty)
            h, z, _, _ = self.rssm.imagine_step(action, h, z)
            reward = self.reward_pred(h, z).item()
            rewards.append(reward)
        return actions, rewards, uncertainties

# Example usage
if __name__ == '__main__':
    agent = WorldModelAgent()
    # Simulate environment
    obs = [0.1, 0.2, 0.3, 0.4]  # Example vector obs
    action = agent.step(obs)
    print("Action:", action)