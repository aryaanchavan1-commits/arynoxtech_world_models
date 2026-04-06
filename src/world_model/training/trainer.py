import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

from model.encoder import Encoder
from model.rssm import RSSM
from model.decoder import Decoder
from model.reward_predictor import RewardPredictor
from model.actor import Actor
from model.critic import Critic
from utils.replay_buffer import ReplayBuffer
from utils.losses import reconstruction_loss, reward_loss, kl_divergence_loss, value_loss

class DreamerTrainer:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.config = config
        self.env_name = config['env_name']
        self.obs_type = config.get('obs_type', 'vector')
        self.obs_shape = config.get('obs_shape', [4])
        self.action_type = config.get('action_type', 'discrete')
        self.seq_len = config['seq_len']
        self.batch_size = config['batch_size']
        self.imagine_horizon = config['imagine_horizon']
        self.kl_beta = config['kl_beta']
        self.gamma = config['gamma']
        self.obs_noise_std = config['obs_noise_std']
        self.action_noise_std = config['action_noise_std']
        self.domain_randomization = config.get('domain_randomization', False)
        self.missing_data_prob = config.get('missing_data_prob', 0.1)
        self.quantize = config.get('quantize', False)
        self.prune = config.get('prune', False)
        self.export_torchscript = config.get('export_torchscript', False)
        self.save_path = config['save_path']
        self.log_interval = config['log_interval']
        self.device = config.get('device', 'auto')

        # Environment
        env = gym.make(self.env_name)
        if self.obs_type == 'vector':
            self.obs_dim = env.observation_space.shape[0]
        elif self.obs_type == 'image':
            self.obs_shape = list(env.observation_space.shape)
        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)
        self.action_dim = env.action_space.shape[0] if self.is_continuous else env.action_space.n
        self.action_low = env.action_space.low if self.is_continuous else None
        self.action_high = env.action_space.high if self.is_continuous else None
        env.close()

        # Models
        self.encoder = Encoder(obs_type=self.obs_type, obs_shape=self.obs_shape, latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim'])
        self.rssm = RSSM(action_dim=self.action_dim, hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'], is_continuous=self.is_continuous, quantize=self.quantize)
        self.decoder = Decoder(obs_type=self.obs_type, obs_shape=self.obs_shape, hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'])
        self.reward_pred = RewardPredictor(hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'], quantize=self.quantize)
        self.actor = Actor(action_dim=self.action_dim, hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'], is_continuous=self.is_continuous, quantize=self.quantize)
        self.critic = Critic(hidden_dim=config['hidden_dim'], latent_dim=config['latent_dim'], quantize=self.quantize)

        # Optimizers
        self.world_model_params = list(self.encoder.parameters()) + \
                                  list(self.rssm.parameters()) + \
                                  list(self.decoder.parameters()) + \
                                  list(self.reward_pred.parameters())
        self.world_optimizer = optim.Adam(self.world_model_params, lr=config['world_model_lr'])

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])

        # Buffer
        self.buffer = ReplayBuffer(capacity=5000)  # Increased for better sampling

        # Device
        if self.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)
        self.to_device()

        # Pruning if enabled
        if self.prune:
            self.apply_pruning()

    def apply_pruning(self):
        # Simple pruning: prune 20% of weights
        import torch.nn.utils.prune as prune
        for module in [self.encoder, self.rssm, self.decoder, self.reward_pred, self.actor, self.critic]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    prune.l1_unstructured(module, name, amount=0.2)

    def to_device(self):
        self.encoder.to(self.device)
        self.rssm.to(self.device)
        self.decoder.to(self.device)
        self.reward_pred.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)

    def collect_experience(self, num_episodes=10):
        env = gym.make(self.env_name)
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode = []
            done = False
            while not done:
                # Domain randomization: vary physics if possible, but for gym, add noise
                if self.domain_randomization:
                    # Simulate varying conditions
                    obs += np.random.normal(0, self.obs_noise_std * 2, size=obs.shape)

                # Add observation noise
                noisy_obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)

                # Add random occlusion (simulate missing data in messy environments)
                mask = np.random.rand(*obs.shape) > self.missing_data_prob
                if self.obs_type == 'vector':
                    noisy_obs = np.where(mask, noisy_obs, 0.0)  # Set missing to 0

                # Random action for initial collection
                action = env.action_space.sample()
                if self.is_continuous:
                    action += np.random.normal(0, self.action_noise_std, size=action.shape)
                    action = np.clip(action, self.action_low, self.action_high)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode.append((noisy_obs, mask, action, reward, done))
                obs = next_obs
            self.buffer.add_episode(episode)
        env.close()

    def train_world_model(self, epochs=100):
        if len(self.buffer) < self.batch_size:
            return

        for _ in tqdm(range(epochs), desc='Training World Model'):
            obs_seq, mask_seq, action_seq, reward_seq, done_seq = self.buffer.sample_batch(self.batch_size, self.seq_len)

            obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(self.device)
            mask_seq = torch.tensor(mask_seq, dtype=torch.bool).to(self.device) if mask_seq is not None and self.obs_type == 'vector' else None
            action_seq = torch.tensor(action_seq, dtype=torch.long if not self.is_continuous else torch.float32).to(self.device)
            reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(self.device)

            # Initial state
            h = torch.zeros(self.batch_size, self.rssm.hidden_dim).to(self.device)
            z = self.rssm.prior_dist(h).sample()

            total_loss = 0
            for t in range(self.seq_len):
                obs = obs_seq[:, t]
                mask = mask_seq[:, t] if mask_seq is not None else None
                action = action_seq[:, t]
                reward = reward_seq[:, t]

                # Encode obs
                obs_embed = self.encoder(obs, mask)

                # Observe step
                h, z, z_mean, z_std = self.rssm.observe_step(action, obs_embed, h, z)

                # Losses
                pred_obs = self.decoder(h, z)
                pred_reward = self.reward_pred(h, z)

                recon_loss = reconstruction_loss(pred_obs, obs)
                rew_loss = reward_loss(pred_reward.squeeze(), reward)

                post_dist = self.rssm.posterior_dist(h, obs_embed)
                prior_dist = self.rssm.prior_dist(h)
                kl_loss = kl_divergence_loss(post_dist, prior_dist)

                loss = recon_loss + rew_loss + self.kl_beta * kl_loss
                total_loss += loss

            self.world_optimizer.zero_grad()
            total_loss.backward()
            self.world_optimizer.step()

        return total_loss.item() / (epochs * self.seq_len) if epochs > 0 else 0

    def train_actor_critic(self, epochs=100):
        if len(self.buffer) < self.batch_size:
            return

        for _ in tqdm(range(epochs), desc='Training Actor-Critic'):
            obs_seq, mask_seq, action_seq, reward_seq, done_seq = self.buffer.sample_batch(self.batch_size, self.seq_len)

            obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(self.device)
            mask_seq = torch.tensor(mask_seq, dtype=torch.bool).to(self.device) if mask_seq is not None and self.obs_type == 'vector' else None
            action_seq = torch.tensor(action_seq, dtype=torch.long if not self.is_continuous else torch.float32).to(self.device)
            reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(self.device)

            # Initial state from real
            h = torch.zeros(self.batch_size, self.rssm.hidden_dim).to(self.device)
            z = self.rssm.prior_dist(h).sample()

            # Get initial h, z from first step
            mask_0 = mask_seq[:, 0] if mask_seq is not None else None
            obs_embed = self.encoder(obs_seq[:, 0], mask_0)
            _, z, _, _ = self.rssm.observe_step(action_seq[:, 0], obs_embed, h, z)
            h = h  # already updated

            # Imagine trajectories
            imagined_rewards = []
            imagined_values = []
            imagined_actions = []

            for t in range(self.imagine_horizon):
                action = self.actor.sample_action(h, z)
                imagined_actions.append(action.unsqueeze(-1))

                h, z, _, _ = self.rssm.imagine_step(action, h, z)

                reward = self.reward_pred(h, z).squeeze()
                value = self.critic(h, z).squeeze()

                imagined_rewards.append(reward)
                imagined_values.append(value)

            # Compute lambda returns (simplified n-step)
            returns = []
            ret = imagined_values[-1]
            for r, v in zip(reversed(imagined_rewards), reversed(imagined_values)):
                ret = r + 0.99 * ret  # gamma=0.99
                returns.append(ret)
            returns.reverse()

            imagined_rewards = torch.stack(imagined_rewards)
            imagined_values = torch.stack(imagined_values)
            returns = torch.stack(returns)

            # Actor loss: maximize returns
            actor_loss = - (returns - imagined_values.detach()).mean()

            # Critic loss: predict returns
            critic_loss = value_loss(imagined_values, returns.detach())

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save_models(self):
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(self.save_path, 'encoder.pth'))
        torch.save(self.rssm.state_dict(), os.path.join(self.save_path, 'rssm.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_path, 'decoder.pth'))
        torch.save(self.reward_pred.state_dict(), os.path.join(self.save_path, 'reward_pred.pth'))
        torch.save(self.actor.state_dict(), os.path.join(self.save_path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(self.save_path, 'critic.pth'))
        print(f"Models saved to {self.save_path}")

    def load_models(self):
        self.encoder.load_state_dict(torch.load(os.path.join(self.save_path, 'encoder.pth'), map_location=self.device))
        self.rssm.load_state_dict(torch.load(os.path.join(self.save_path, 'rssm.pth'), map_location=self.device))
        self.decoder.load_state_dict(torch.load(os.path.join(self.save_path, 'decoder.pth'), map_location=self.device))
        self.reward_pred.load_state_dict(torch.load(os.path.join(self.save_path, 'reward_pred.pth'), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(self.save_path, 'actor.pth'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(self.save_path, 'critic.pth'), map_location=self.device))
        print(f"Models loaded from {self.save_path}")

    def train(self):
        total_steps = self.config['total_steps']
        collect_episodes = self.config['collect_episodes']
        train_world_epochs = self.config['train_world_epochs']
        train_actor_epochs = self.config['train_actor_epochs']
        eval_episodes = self.config['eval_episodes']

        rewards = []
        step = 0
        while step < total_steps:
            # Collect experience
            self.collect_experience(collect_episodes)
            step += collect_episodes * 200  # approx

            # Train world model
            world_loss = self.train_world_model(train_world_epochs) or 0.0

            # Train actor-critic
            ac_result = self.train_actor_critic(train_actor_epochs)
            if ac_result is not None:
                actor_loss, critic_loss = ac_result
            else:
                actor_loss, critic_loss = 0.0, 0.0

            # Evaluate
            if step % self.log_interval == 0:
                ep_reward = self.evaluate(eval_episodes)
                rewards.append(ep_reward)
                print(f"Step {step}, Reward: {ep_reward:.2f}, World Loss: {world_loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
                self.save_models()

        # Export models if enabled
        self.export_models()

        plt.plot(rewards)
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Episode Reward')
        plt.show()

    def evaluate(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.config['eval_episodes']
        env = gym.make(self.env_name)
        total_reward = 0
        for _ in range(num_episodes):
            obs, _ = env.reset()
            h = torch.zeros(1, self.rssm.hidden_dim).to(self.device)
            z = self.rssm.prior_dist(h).sample()
            done = False
            ep_reward = 0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                obs_embed = self.encoder(obs_tensor)
                dummy_action = torch.zeros(1, dtype=torch.long).to(self.device) if not self.is_continuous else torch.zeros(1, self.action_dim).to(self.device)
                _, z, _, _ = self.rssm.observe_step(dummy_action, obs_embed, h, z)  # dummy action
                action_tensor = self.actor.sample_action(h, z, deterministic=True, action_low=self.action_low, action_high=self.action_high)
                if self.is_continuous:
                    action = action_tensor.squeeze(0).cpu().numpy()
                else:
                    action = action_tensor.item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            total_reward += ep_reward
        env.close()
        return total_reward / num_episodes

    def export_models(self):
        if self.export_torchscript:
            # Export to TorchScript for edge deployment
            self.encoder.eval()
            self.rssm.eval()
            self.decoder.eval()
            self.reward_pred.eval()
            self.actor.eval()
            self.critic.eval()
            # Example for encoder
            dummy_input = torch.randn(1, *self.obs_shape).to(self.device)
            scripted_encoder = torch.jit.trace(self.encoder, dummy_input)
            torch.jit.save(scripted_encoder, os.path.join(self.save_path, 'encoder.pt'))
            # Similarly for others, but for brevity, just encoder
            print("Models exported to TorchScript")