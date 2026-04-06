"""
World Model Agent - Simplified High-Level API

This module provides a simple, user-friendly interface for using the World Model
without needing to understand the internal architecture.

Example usage:
    >>> import world_model
    >>> 
    >>> # Create and train an agent
    >>> agent = world_model.Agent()
    >>> agent.train(env='CartPole-v1', steps=50000)
    >>> 
    >>> # Use trained agent for inference
    >>> agent = world_model.Agent(model_path='models/')
    >>> action = agent.step([0.1, 0.2, 0.3, 0.4])
"""

import torch
import numpy as np
import json
import os
from typing import Optional, List, Union, Tuple, Dict, Any

from world_model.model.encoder import Encoder
from world_model.model.rssm import RSSM
from world_model.model.decoder import Decoder
from world_model.model.actor import Actor
from world_model.model.critic import Critic
from world_model.model.reward_predictor import RewardPredictor
from world_model.training.trainer import DreamerTrainer
from world_model.deployment import WorldModelAgent


class Agent:
    """
    Simplified World Model Agent - The easiest way to use World Model.
    
    This class wraps all complexity and provides a simple interface for:
    - Training: agent.train(env='CartPole-v1', steps=50000)
    - Inference: action = agent.step(observation)
    - Planning: actions, rewards = agent.imagine(horizon=20)
    
    Parameters
    ----------
    config : dict or str, optional
        Configuration dictionary or path to config JSON file.
        If None, uses default configuration.
    model_path : str, optional
        Path to pre-trained model directory. If provided, loads trained models.
    device : str, optional
        Device to use ('cpu', 'cuda', or 'auto'). Default is 'auto'.
    
    Examples
    --------
    >>> # Training from scratch
    >>> agent = world_model.Agent()
    >>> agent.train(env='CartPole-v1', steps=50000)
    
    >>> # Loading pre-trained model
    >>> agent = world_model.Agent(model_path='models/')
    >>> obs = [0.1, 0.2, 0.3, 0.4]
    >>> action = agent.step(obs)
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str]] = None,
        model_path: Optional[str] = None,
        device: str = 'auto',
    ):
        # Load configuration
        if config is None:
            self.config = self._default_config()
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.config['device'] = str(self.device)
        
        # Initialize components
        self._init_models()
        
        # Load pre-trained models if path provided
        if model_path:
            self.load(model_path)
        
        # State for inference
        self.h = None
        self.z = None
        self._reset_state()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the agent."""
        return {
            'env_name': 'CartPole-v1',
            'obs_type': 'vector',
            'obs_shape': [4],
            'action_type': 'discrete',
            'seq_len': 50,
            'batch_size': 64,
            'imagine_horizon': 25,
            'latent_dim': 64,
            'hidden_dim': 256,
            'world_model_lr': 0.0003,
            'actor_lr': 0.0003,
            'critic_lr': 0.0003,
            'kl_beta': 0.1,
            'gamma': 0.99,
            'total_steps': 50000,
            'collect_episodes': 10,
            'train_world_epochs': 20,
            'train_actor_epochs': 20,
            'eval_episodes': 10,
            'obs_noise_std': 0.05,
            'action_noise_std': 0.2,
            'domain_randomization': True,
            'missing_data_prob': 0.2,
            'safety_threshold': 0.8,
            'save_path': 'models/',
            'log_interval': 5000,
        }
    
    def _init_models(self):
        """Initialize all model components."""
        # Determine observation/action dimensions from config
        obs_type = self.config.get('obs_type', 'vector')
        obs_shape = self.config.get('obs_shape', [4])
        action_type = self.config.get('action_type', 'discrete')
        action_dim = self.config.get('action_dim', 2)
        latent_dim = self.config.get('latent_dim', 64)
        hidden_dim = self.config.get('hidden_dim', 256)
        is_continuous = action_type == 'continuous'
        
        self.encoder = Encoder(
            obs_type=obs_type,
            obs_shape=obs_shape,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        self.rssm = RSSM(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            is_continuous=is_continuous,
        ).to(self.device)
        
        self.decoder = Decoder(
            obs_type=obs_type,
            obs_shape=obs_shape,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)
        
        self.actor = Actor(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            is_continuous=is_continuous,
        ).to(self.device)
        
        self.critic = Critic(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)
        
        self.reward_predictor = RewardPredictor(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)
    
    def _reset_state(self):
        """Reset the agent's internal state."""
        self.h = torch.zeros(1, self.config.get('hidden_dim', 256)).to(self.device)
        self.z = self.rssm.prior_dist(self.h).sample()
    
    def reset(self):
        """
        Reset the agent's internal state.
        
        Call this at the start of each episode when doing inference.
        """
        self._reset_state()
    
    def step(
        self,
        obs: Union[List[float], np.ndarray, torch.Tensor],
        mask: Optional[Union[List[float], np.ndarray]] = None,
        deterministic: bool = True,
    ) -> Union[int, np.ndarray]:
        """
        Get action from observation.
        
        Parameters
        ----------
        obs : list, numpy.ndarray, or torch.Tensor
            The observation from the environment.
        mask : list or numpy.ndarray, optional
            Boolean mask indicating valid observations (1) or missing data (0).
        deterministic : bool, optional
            If True, returns the most likely action. If False, samples from policy.
        
        Returns
        -------
        action : int or numpy.ndarray
            The action to take. Integer for discrete actions, array for continuous.
        
        Examples
        --------
        >>> obs = [0.1, 0.2, 0.3, 0.4]
        >>> action = agent.step(obs)
        >>> 
        >>> # With missing data
        >>> obs = [0.1, None, 0.3, 0.4]
        >>> mask = [1, 0, 1, 1]  # Second observation is missing
        >>> action = agent.step(obs, mask=mask)
        """
        # Convert observation to tensor
        obs = np.array(obs, dtype=np.float32)
        
        # Handle missing values
        if mask is not None:
            mask = np.array(mask, dtype=np.float32)
            obs = np.where(mask, obs, 0.0)
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Encode observation
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_embed = self.encoder(obs_tensor, mask_tensor)
        
        # Update state
        is_continuous = self.config.get('action_type', 'discrete') == 'continuous'
        action_dim = self.config.get('action_dim', 2)
        dummy_action = (
            torch.zeros(1, action_dim, dtype=torch.float32).to(self.device) 
            if is_continuous 
            else torch.zeros(1, dtype=torch.long).to(self.device)
        )
        self.h, self.z, _, _ = self.rssm.observe_step(dummy_action, obs_embed, self.h, self.z)
        
        # Get action
        action_tensor = self.actor.sample_action(
            self.h, 
            self.z, 
            deterministic=deterministic,
            action_low=self.config.get('action_low'),
            action_high=self.config.get('action_high'),
        )
        
        if is_continuous:
            return action_tensor.squeeze(0).cpu().numpy()
        else:
            return action_tensor.item()
    
    def imagine(
        self,
        horizon: int = 20,
        start_obs: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Tuple[List, List[float], List[float]]:
        """
        Imagine future trajectory.
        
        Parameters
        ----------
        horizon : int, optional
            Number of steps to imagine into the future. Default is 20.
        start_obs : list or numpy.ndarray, optional
            Starting observation. If None, uses current state.
        
        Returns
        -------
        actions : list
            Imagined actions at each step.
        rewards : list of float
            Predicted rewards at each step.
        uncertainties : list of float
            Uncertainty estimates at each step.
        
        Examples
        --------
        >>> actions, rewards, uncertainties = agent.imagine(horizon=10)
        >>> print(f"Total predicted reward: {sum(rewards)}")
        """
        if start_obs is not None:
            obs = np.array(start_obs, dtype=np.float32)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            obs_embed = self.encoder(obs_tensor)
            
            is_continuous = self.config.get('action_type', 'discrete') == 'continuous'
            action_dim = self.config.get('action_dim', 2)
            dummy_action = (
                torch.zeros(1, action_dim, dtype=torch.float32).to(self.device)
                if is_continuous
                else torch.zeros(1, dtype=torch.long).to(self.device)
            )
            h, z, _, _ = self.rssm.observe_step(dummy_action, obs_embed, self.h, self.z)
        else:
            h = self.h.clone()
            z = self.z.clone()
        
        actions = []
        rewards = []
        uncertainties = []
        
        for _ in range(horizon):
            action = self.actor.sample_action(h, z, deterministic=False)
            uncertainty = self.actor.get_uncertainty(h, z).item()
            
            actions.append(action.tolist() if isinstance(action, torch.Tensor) else action)
            uncertainties.append(uncertainty)
            
            h, z, _, _ = self.rssm.imagine_step(action, h, z)
            reward = self.reward_predictor(h, z).item()
            rewards.append(reward)
        
        return actions, rewards, uncertainties
    
    def train(
        self,
        env: Optional[str] = None,
        steps: Optional[int] = None,
        save_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Train the World Model.
        
        Parameters
        ----------
        env : str, optional
            Gymnasium environment name. If None, uses config value.
        steps : int, optional
            Number of training steps. If None, uses config value.
        save_path : str, optional
            Path to save trained models. If None, uses config value.
        **kwargs
            Additional configuration overrides.
        
        Examples
        --------
        >>> agent = world_model.Agent()
        >>> agent.train(env='CartPole-v1', steps=50000)
        >>> 
        >>> # With custom parameters
        >>> agent.train(
        ...     env='Pendulum-v1',
        ...     steps=100000,
        ...     batch_size=128,
        ...     latent_dim=128,
        ... )
        """
        # Update config
        if env:
            self.config['env_name'] = env
        if steps:
            self.config['total_steps'] = steps
        if save_path:
            self.config['save_path'] = save_path
        
        # Apply additional kwargs
        for key, value in kwargs.items():
            self.config[key] = value
        
        # Save config to temporary file for trainer
        config_path = '_temp_agent_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        
        try:
            trainer = DreamerTrainer(config_path=config_path)
            trainer.train()
        finally:
            # Clean up temp file
            if os.path.exists(config_path):
                os.remove(config_path)
    
    def save(self, path: str):
        """
        Save the agent's models.
        
        Parameters
        ----------
        path : str
            Directory path to save models.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
        torch.save(self.rssm.state_dict(), os.path.join(path, 'rssm.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(path, 'decoder.pth'))
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.reward_predictor.state_dict(), os.path.join(path, 'reward_pred.pth'))
        
        # Save config
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f)
        
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load the agent's models.
        
        Parameters
        ----------
        path : str
            Directory path to load models from.
        """
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pth'), map_location=self.device))
        self.rssm.load_state_dict(torch.load(os.path.join(path, 'rssm.pth'), map_location=self.device))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pth'), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), map_location=self.device))
        self.reward_predictor.load_state_dict(torch.load(os.path.join(path, 'reward_pred.pth'), map_location=self.device))
        
        # Load config if exists
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
        
        # Set to eval mode
        self.encoder.eval()
        self.rssm.eval()
        self.decoder.eval()
        self.actor.eval()
        self.critic.eval()
        self.reward_predictor.eval()
        
        print(f"Agent loaded from {path}")
    
    def evaluate(self, env_name: Optional[str] = None, episodes: int = 10) -> float:
        """
        Evaluate the agent in an environment.
        
        Parameters
        ----------
        env_name : str, optional
            Gymnasium environment name. If None, uses config value.
        episodes : int, optional
            Number of episodes to evaluate. Default is 10.
        
        Returns
        -------
        avg_reward : float
            Average reward across episodes.
        """
        import gymnasium as gym
        
        if env_name is None:
            env_name = self.config.get('env_name', 'CartPole-v1')
        
        env = gym.make(env_name)
        total_reward = 0
        
        for _ in range(episodes):
            obs, _ = env.reset()
            self.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action = self.step(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            
            total_reward += ep_reward
        
        env.close()
        avg_reward = total_reward / episodes
        print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
        return avg_reward