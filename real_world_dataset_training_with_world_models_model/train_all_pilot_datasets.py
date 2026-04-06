#!/usr/bin/env python3
"""
Universal Training Script for ALL Pilot Datasets.
Trains world model on Smart Factory IoT, NASA Turbofan, and Bearing Faults.

Usage:
    python train_all_pilot_datasets.py --dataset smart_factory
    python train_all_pilot_datasets.py --dataset nasa_turbofan
    python train_all_pilot_datasets.py --dataset bearing_faults
    python train_all_pilot_datasets.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'model'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'utils'))

from model.encoder import Encoder
from model.rssm import RSSM
from model.decoder import Decoder
from model.reward_predictor import RewardPredictor
from model.actor import Actor
from model.critic import Critic
from utils.replay_buffer import ReplayBuffer
from utils.losses import reconstruction_loss, reward_loss, kl_divergence_loss

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Dataset configurations
DATASET_CONFIGS = {
    'smart_factory': {
        'data_path': os.path.join(DATA_DIR, 'smart_factory', 'smart_factory_sensor_data.csv'),
        'obs_dim': 52,
        'sensor_columns': [f'sensor_{i}' for i in range(52)],
        'failure_column': None,
        'seq_len': 50,
        'batch_size': 32,
        'epochs': 50,
        'label': 'Smart Factory IoT - 52 Sensors',
    },
    'nasa_turbofan': {
        'data_path': os.path.join(DATA_DIR, 'nasa_turbofan', 'nasa_turbofan_1gb.csv'),
        'obs_dim': 24,
        'sensor_columns': ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(3)] + [f'sensor_{i}' for i in range(21)],
        'failure_column': None,
        'seq_len': 50,
        'batch_size': 32,
        'epochs': 50,
        'label': 'NASA Turbofan Engine - 24 Sensors',
    },
    'bearing_faults': {
        'data_path': os.path.join(DATA_DIR, 'bearing_faults', 'bearing_faults_1gb.csv'),
        'obs_dim': 6,
        'sensor_columns': ['vibration_x', 'vibration_y', 'vibration_z', 'temperature', 'rpm', 'load'],
        'failure_column': 'severity',
        'seq_len': 50,
        'batch_size': 32,
        'epochs': 50,
        'label': 'Bearing Faults - 6 Vibration Sensors',
    }
}

def load_pilot_dataset(config):
    """Load pilot dataset and create episodes."""
    data_path = config['data_path']
    sensor_columns = config['sensor_columns']
    seq_len = config['seq_len']

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, nrows=5_000_000)

    available_columns = [c for c in sensor_columns if c in df.columns]
    if len(available_columns) < len(sensor_columns):
        print(f"  ⚠️ Some columns missing, using {len(available_columns)} available columns")

    sensor_data = df[available_columns].values.astype(np.float32)
    obs_dim = sensor_data.shape[1]

    # Normalize
    mean = sensor_data.mean(axis=0)
    std = sensor_data.std(axis=0) + 1e-8
    sensor_data = (sensor_data - mean) / std

    print(f"  Loaded {len(sensor_data)} samples, {obs_dim} sensors")

    # Create episodes
    episodes = []
    overlap = seq_len // 2
    for start in range(0, len(sensor_data) - seq_len, seq_len - overlap):
        episode = []
        for t in range(start, start + seq_len):
            obs = sensor_data[t]
            action = np.array([0.0])
            reward = 1.0
            mask = np.ones(obs_dim, dtype=np.float32)
            done = (t == start + seq_len - 1)
            episode.append((obs, mask, action, reward, done))
        episodes.append(episode)

    print(f"  Created {len(episodes)} episodes")
    return episodes, obs_dim, sensor_data

def initialize_models(obs_dim, device):
    """Initialize all model components."""
    hidden_dim = 128
    latent_dim = 64
    action_dim = 1

    encoder = Encoder(obs_type='vector', obs_shape=[obs_dim], latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    rssm = RSSM(action_dim=action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, is_continuous=False).to(device)
    decoder = Decoder(obs_type='vector', obs_shape=[obs_dim], hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    reward_pred = RewardPredictor(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    actor = Actor(action_dim=action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, is_continuous=False).to(device)
    critic = Critic(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in rssm.parameters()) + \
                   sum(p.numel() for p in decoder.parameters()) + \
                   sum(p.numel() for p in reward_pred.parameters())
    print(f"  Total model params: {total_params:,}")

    return encoder, rssm, decoder, reward_pred, actor, critic

def train_epoch(encoder, rssm, decoder, reward_pred, buffer, batch_size, seq_len, kl_beta, device):
    """Train one epoch."""
    if len(buffer) < batch_size:
        return None

    obs_seq, mask_seq, action_seq, reward_seq, done_seq = buffer.sample_batch(batch_size, seq_len)
    obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(device)
    mask_seq = torch.tensor(mask_seq, dtype=torch.bool).to(device)
    action_seq = torch.tensor(action_seq, dtype=torch.long).squeeze(-1).to(device)
    reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(device)

    world_params = list(encoder.parameters()) + list(rssm.parameters()) + \
                   list(decoder.parameters()) + list(reward_pred.parameters())
    optimizer = torch.optim.Adam(world_params, lr=0.0003)

    h = torch.zeros(batch_size, rssm.hidden_dim).to(device)
    z = rssm.prior_dist(h).sample()

    total_loss = 0
    for t in range(seq_len):
        obs = obs_seq[:, t]
        mask = mask_seq[:, t]
        action = action_seq[:, t]
        reward = reward_seq[:, t]

        obs_embed = encoder(obs, mask)
        h, z, _, _ = rssm.observe_step(action, obs_embed, h, z)
        pred_obs = decoder(h, z)
        pred_reward = reward_pred(h, z)

        r_loss = reconstruction_loss(pred_obs, obs)
        rew_loss = reward_loss(pred_reward.squeeze(), reward)
        post_dist = rssm.posterior_dist(h, obs_embed)
        prior_dist = rssm.prior_dist(h)
        k_loss = kl_divergence_loss(post_dist, prior_dist)

        total_loss += r_loss + rew_loss + kl_beta * k_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(world_params, max_norm=1.0)
    optimizer.step()

    return total_loss.item() / seq_len

def evaluate(encoder, rssm, decoder, reward_pred, sensor_data, device):
    """Evaluate reconstruction quality."""
    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_pred.eval()

    h = torch.zeros(1, rssm.hidden_dim).to(device)
    z = rssm.prior_dist(h).sample()

    errors = []
    with torch.no_grad():
        indices = np.random.choice(len(sensor_data), min(5000, len(sensor_data)), replace=False)
        for idx in indices:
            obs = torch.tensor(sensor_data[idx:idx+1], dtype=torch.float32).to(device)
            mask = torch.ones_like(obs).bool()
            obs_embed = encoder(obs, mask)
            h, z, _, _ = rssm.observe_step(torch.zeros(1, dtype=torch.long).to(device), obs_embed, h, z)
            pred_obs = decoder(h, z)
            error = ((pred_obs - obs) ** 2).mean().item()
            errors.append(error)

    encoder.train()
    rssm.train()
    decoder.train()
    reward_pred.train()

    return np.mean(errors), np.std(errors), np.percentile(errors, 95)

def save_models(encoder, rssm, decoder, reward_pred, actor, critic, save_dir):
    """Save models."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder.pth'))
    torch.save(rssm.state_dict(), os.path.join(save_dir, 'rssm.pth'))
    torch.save(decoder.state_dict(), os.path.join(save_dir, 'decoder.pth'))
    torch.save(reward_pred.state_dict(), os.path.join(save_dir, 'reward_pred.pth'))
    torch.save(actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
    torch.save(critic.state_dict(), os.path.join(save_dir, 'critic.pth'))

def plot_training(history, dataset_name, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['total_loss'], 'b-')
    axes[0].set_title(f'{dataset_name} - Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[1].plot(history['eval_error'], 'g-')
    axes[1].set_title(f'{dataset_name} - Eval Reconstruction Error')
    axes[1].set_xlabel('Eval Step')
    axes[1].set_ylabel('Error')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def train_dataset(dataset_name):
    """Train world model on a single dataset."""
    config = DATASET_CONFIGS[dataset_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*70)
    print(f"🚀 TRAINING WORLD MODEL: {config['label']}")
    print("="*70)
    print(f"  Device: {device}")
    print(f"  Data: {config['data_path']}")
    print(f"  Sensors: {config['obs_dim']}")
    print(f"  Epochs: {config['epochs']}")

    # Load data
    episodes, obs_dim, sensor_data = load_pilot_dataset(config)

    # Initialize models
    encoder, rssm, decoder, reward_pred, actor, critic = initialize_models(obs_dim, device)

    # Fill replay buffer
    buffer = ReplayBuffer(capacity=50000)
    for ep in tqdm(episodes, desc="Adding to buffer"):
        buffer.add_episode(ep)
    print(f"  Buffer: {len(buffer)} episodes")

    # Training loop
    history = {'total_loss': [], 'eval_error': []}
    best_loss = float('inf')

    for epoch in range(config['epochs']):
        loss = train_epoch(encoder, rssm, decoder, reward_pred, buffer,
                          config['batch_size'], config['seq_len'], 0.1, device)
        if loss is None:
            continue

        history['total_loss'].append(loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            mean_err, std_err, p95_err = evaluate(encoder, rssm, decoder, reward_pred, sensor_data, device)
            history['eval_error'].append(mean_err)
            print(f"\n  Epoch {epoch+1}/{config['epochs']}: Loss={loss:.4f}, Eval={mean_err:.4f}, P95={p95_err:.4f}")

            if loss < best_loss:
                best_loss = loss
                save_dir = os.path.join(MODELS_DIR, f'{dataset_name}')
                save_models(encoder, rssm, decoder, reward_pred, actor, critic, save_dir)
                print(f"  ✅ Best model saved to {save_dir}/")

    # Final evaluation
    mean_err, std_err, p95_err = evaluate(encoder, rssm, decoder, reward_pred, sensor_data, device)

    # Save report
    report = {
        'dataset': dataset_name,
        'samples': len(sensor_data),
        'episodes': len(episodes),
        'epochs': config['epochs'],
        'obs_dim': obs_dim,
        'best_loss': best_loss,
        'final_recon_error': mean_err,
        'p95_recon_error': p95_err,
    }

    report_path = os.path.join(MODELS_DIR, f'{dataset_name}_training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Plot
    plot_path = os.path.join(MODELS_DIR, f'{dataset_name}_training.png')
    plot_training(history, config['label'], plot_path)

    print(f"\n✅ {dataset_name} Training Complete!")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Recon Error: {mean_err:.4f} ± {std_err:.4f}")
    print(f"  Report: {report_path}")
    print(f"  Plot: {plot_path}")

    return report

def main():
    parser = argparse.ArgumentParser(description='Train World Model on Pilot Datasets')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'smart_factory', 'nasa_turbofan', 'bearing_faults'],
                       help='Dataset to train on')
    args = parser.parse_args()

    datasets = ['smart_factory', 'nasa_turbofan', 'bearing_faults'] if args.dataset == 'all' else [args.dataset]

    results = {}
    for dataset_name in datasets:
        report = train_dataset(dataset_name)
        results[dataset_name] = report

    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY - ALL DATASETS")
    print("="*70)
    for name, report in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Samples: {report['samples']:,}")
        print(f"  Epochs: {report['epochs']}")
        print(f"  Best Loss: {report['best_loss']:.4f}")
        print(f"  Recon Error: {report['final_recon_error']:.4f}")
        print(f"  P95 Error: {report['p95_recon_error']:.4f}")

    print("\n🎯 All models trained and saved!")
    print("  Run: python evaluate_all_pilot_datasets.py")

if __name__ == '__main__':
    main()