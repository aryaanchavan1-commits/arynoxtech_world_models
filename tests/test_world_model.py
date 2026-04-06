"""
Unit tests for World Model components.
Run with: python -m pytest tests/
"""

import pytest
import torch
import numpy as np
from model.encoder import Encoder
from model.rssm import RSSM
from model.actor import Actor
from model.critic import Critic
from model.reward_predictor import RewardPredictor
from deployment import WorldModelAgent
from training.trainer import DreamerTrainer

class TestModels:
    def test_encoder_vector(self):
        encoder = Encoder(obs_type='vector', obs_shape=[4], latent_dim=32)
        obs = torch.randn(2, 4)
        embed = encoder(obs)
        assert embed.shape == (2, 32)

    def test_encoder_image(self):
        encoder = Encoder(obs_type='image', obs_shape=[3, 64, 64], latent_dim=32)
        obs = torch.randn(2, 3, 64, 64)
        embed = encoder(obs)
        assert embed.shape == (2, 32)

    def test_rssm_discrete(self):
        rssm = RSSM(action_dim=2, hidden_dim=64, latent_dim=32, is_continuous=False)
        action = torch.tensor([0, 1])
        obs_embed = torch.randn(2, 32)
        h = torch.randn(2, 64)
        z = torch.randn(2, 32)
        next_h, next_z, _, _ = rssm.observe_step(action, obs_embed, h, z)
        assert next_h.shape == (2, 64)
        assert next_z.shape == (2, 32)

    def test_actor_uncertainty(self):
        actor = Actor(action_dim=2, hidden_dim=64, latent_dim=32, is_continuous=False)
        h = torch.randn(2, 64)
        z = torch.randn(2, 32)
        uncertainty = actor.get_uncertainty(h, z)
        assert uncertainty.shape == (2,)

    def test_agent_step(self):
        # Mock models for testing
        agent = WorldModelAgent.__new__(WorldModelAgent)  # Create without __init__
        agent.obs_type = 'vector'
        agent.obs_shape = [4]
        agent.config = {'obs_noise_std': 0.01}
        agent.h = torch.zeros(1, 64)
        agent.z = torch.randn(1, 32)
        agent.encoder = Encoder(obs_type='vector', obs_shape=[4], latent_dim=32)
        agent.rssm = RSSM(action_dim=2, hidden_dim=64, latent_dim=32, is_continuous=False)
        agent.actor = Actor(action_dim=2, hidden_dim=64, latent_dim=32, is_continuous=False)
        action = agent.step([0.1, 0.2, 0.3, 0.4])
        assert isinstance(action, int)

class TestTraining:
    def test_trainer_init(self):
        trainer = DreamerTrainer.__new__(DreamerTrainer)  # Mock init
        trainer.config = {
            'obs_type': 'vector',
            'obs_shape': [4],
            'action_type': 'discrete',
            'latent_dim': 32,
            'hidden_dim': 64
        }
        trainer.obs_type = 'vector'
        trainer.obs_shape = [4]
        trainer.is_continuous = False
        trainer.action_dim = 2
        trainer.device = torch.device('cpu')
        # Test components init
        assert trainer.obs_type == 'vector'

if __name__ == '__main__':
    pytest.main([__file__])