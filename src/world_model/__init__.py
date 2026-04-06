"""
World Model - Advanced DreamerV3 with RSSM for Universal Industrial and Real-World Applications

A comprehensive World Model implementation for reinforcement learning, designed for
industrial automation, robotics, drones, autonomous vehicles, and intelligent systems.

Example usage:
    >>> import world_model
    >>> 
    >>> # Quick start with trainer
    >>> trainer = world_model.DreamerTrainer()
    >>> trainer.train()
    >>> 
    >>> # Or use the agent for inference
    >>> agent = world_model.Agent(model_path='models/')
    >>> action = agent.step([0.1, 0.2, 0.3, 0.4])
"""

__version__ = "1.0.0"
__author__ = "Aryan Sanjay Chavan"
__email__ = "aryaanchavan1@gmail.com"
__license__ = "MIT"

# Model components
from world_model.model.encoder import Encoder
from world_model.model.rssm import RSSM
from world_model.model.decoder import Decoder
from world_model.model.actor import Actor
from world_model.model.critic import Critic
from world_model.model.reward_predictor import RewardPredictor

# Training
from world_model.training.trainer import DreamerTrainer

# Deployment
from world_model.deployment import WorldModelAgent

# Utilities
from world_model.utils.replay_buffer import ReplayBuffer

# High-level API
from world_model.agent import Agent


__all__ = [
    # Core model components
    "Encoder",
    "RSSM",
    "Decoder",
    "Actor",
    "Critic",
    "RewardPredictor",
    
    # Training
    "DreamerTrainer",
    
    # Deployment
    "WorldModelAgent",
    "Agent",
    
    # Utilities
    "ReplayBuffer",
    
    # Metadata
    "__version__",
    "__author__",
]