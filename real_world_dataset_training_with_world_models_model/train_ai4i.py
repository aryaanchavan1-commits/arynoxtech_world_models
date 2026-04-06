#!/usr/bin/env python3
"""
AI4I Predictive Maintenance - World Model Training Script
Trains a world model on industrial sensor data for anomaly detection.

Usage:
    python train_ai4i.py                    # Download + Train + Evaluate
    python train_ai4i.py --skip-download   # Train only
    python train_ai4i.py --download-only   # Download only
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from model.encoder import Encoder
from model.rssm import RSSM
from model.decoder import Decoder
from model.reward_predictor import RewardPredictor
from model.actor import Actor
from model.critic import Critic
from utils.replay_buffer import ReplayBuffer
from utils.losses import reconstruction_loss, reward_loss, kl_divergence_loss

# Import data loader from local folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders.ai4i_predictive import AI4IDatasetLoader, load_ai4i_for_training

def setup_device(config):
    """Setup compute device."""
    if config.get('device', 'auto') == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    print(f"  Device: {device}")
    return device

def initialize_models(config, device):
    """Initialize all model components."""
    obs_shape = config.get('obs_shape', [5])
    obs_dim = obs_shape[0]
    action_dim = config.get('action_dim', 1)
    hidden_dim = config.get('hidden_dim', 128)
    latent_dim = config.get('latent_dim', 64)
    is_continuous = config.get('action_type', 'discrete') == 'continuous'

    print("\n🤖 Initializing models...")

    encoder = Encoder(
        obs_type='vector',
        obs_shape=obs_shape,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)

    rssm = RSSM(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        is_continuous=is_continuous
    ).to(device)

    decoder = Decoder(
        obs_type='vector',
        obs_shape=obs_shape,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)

    reward_pred = RewardPredictor(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)

    actor = Actor(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        is_continuous=is_continuous
    ).to(device)

    critic = Critic(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in rssm.parameters()) + \
                   sum(p.numel() for p in decoder.parameters()) + \
                   sum(p.numel() for p in reward_pred.parameters()) + \
                   sum(p.numel() for p in actor.parameters()) + \
                   sum(p.numel() for p in critic.parameters())

    print(f"  Encoder:     {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  RSSM:        {sum(p.numel() for p in rssm.parameters()):,} params")
    print(f"  Decoder:     {sum(p.numel() for p in decoder.parameters()):,} params")
    print(f"  RewardPred:  {sum(p.numel() for p in reward_pred.parameters()):,} params")
    print(f"  Actor:       {sum(p.numel() for p in actor.parameters()):,} params")
    print(f"  Critic:      {sum(p.numel() for p in critic.parameters()):,} params")
    print(f"  Total:       {total_params:,} params")

    return encoder, rssm, decoder, reward_pred, actor, critic

def train_world_model_epoch(encoder, rssm, decoder, reward_pred, buffer, config, device):
    """Train world model for one epoch."""
    batch_size = config.get('batch_size', 32)
    seq_len = config.get('seq_len', 50)
    kl_beta = config.get('kl_beta', 0.1)

    if len(buffer) < batch_size:
        return None

    # Sample batch
    obs_seq, mask_seq, action_seq, reward_seq, done_seq = buffer.sample_batch(batch_size, seq_len)

    obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(device)
    mask_seq = torch.tensor(mask_seq, dtype=torch.bool).to(device)
    action_seq = torch.tensor(action_seq, dtype=torch.long).squeeze(-1).to(device)
    reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(device)

    # Optimizer
    world_params = list(encoder.parameters()) + list(rssm.parameters()) + \
                   list(decoder.parameters()) + list(reward_pred.parameters())
    optimizer = torch.optim.Adam(world_params, lr=config.get('world_model_lr', 0.0003))

    # Forward pass
    h = torch.zeros(batch_size, rssm.hidden_dim).to(device)
    z = rssm.prior_dist(h).sample()

    total_loss = 0
    recon_losses = []
    reward_losses = []
    kl_losses = []

    for t in range(seq_len):
        obs = obs_seq[:, t]
        mask = mask_seq[:, t]
        action = action_seq[:, t]
        reward = reward_seq[:, t]

        # Encode
        obs_embed = encoder(obs, mask)

        # Observe step
        h, z, z_mean, z_std = rssm.observe_step(action, obs_embed, h, z)

        # Decode observation
        pred_obs = decoder(h, z)

        # Predict reward
        pred_reward = reward_pred(h, z)

        # Compute losses
        r_loss = reconstruction_loss(pred_obs, obs)
        rew_loss = reward_loss(pred_reward.squeeze(), reward)

        post_dist = rssm.posterior_dist(h, obs_embed)
        prior_dist = rssm.prior_dist(h)
        k_loss = kl_divergence_loss(post_dist, prior_dist)

        step_loss = r_loss + rew_loss + kl_beta * k_loss
        total_loss += step_loss

        recon_losses.append(r_loss.item())
        reward_losses.append(rew_loss.item())
        kl_losses.append(k_loss.item())

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(world_params, max_norm=1.0)
    optimizer.step()

    return {
        'total_loss': total_loss.item() / seq_len,
        'recon_loss': np.mean(recon_losses),
        'reward_loss': np.mean(reward_losses),
        'kl_loss': np.mean(kl_losses)
    }

def evaluate_model(encoder, rssm, decoder, reward_pred, buffer, config, device):
    """Evaluate reconstruction quality."""
    batch_size = min(10, config.get('batch_size', 32))
    seq_len = config.get('seq_len', 50)

    if len(buffer) < batch_size:
        return None

    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_pred.eval()

    with torch.no_grad():
        obs_seq, mask_seq, action_seq, reward_seq, done_seq = buffer.sample_batch(batch_size, seq_len)

        obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(device)
        action_seq = torch.tensor(action_seq, dtype=torch.long).squeeze(-1).to(device)
        reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(device)

        h = torch.zeros(batch_size, rssm.hidden_dim).to(device)
        z = rssm.prior_dist(h).sample()

        recon_errors = []
        reward_errors = []

        for t in range(seq_len):
            obs = obs_seq[:, t]
            mask = torch.ones_like(obs).bool()

            obs_embed = encoder(obs, mask)
            h, z, _, _ = rssm.observe_step(action_seq[:, t], obs_embed, h, z)

            pred_obs = decoder(h, z)
            pred_reward = reward_pred(h, z)

            recon_error = ((pred_obs - obs) ** 2).mean().item()
            reward_error = ((pred_reward.squeeze() - reward_seq[:, t]) ** 2).mean().item()

            recon_errors.append(recon_error)
            reward_errors.append(reward_error)

    encoder.train()
    rssm.train()
    decoder.train()
    reward_pred.train()

    return {
        'recon_error': np.mean(recon_errors),
        'reward_error': np.mean(reward_errors)
    }

def save_models(encoder, rssm, decoder, reward_pred, actor, critic, save_path):
    """Save all model weights."""
    os.makedirs(save_path, exist_ok=True)

    torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder.pth'))
    torch.save(rssm.state_dict(), os.path.join(save_path, 'rssm.pth'))
    torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.pth'))
    torch.save(reward_pred.state_dict(), os.path.join(save_path, 'reward_pred.pth'))
    torch.save(actor.state_dict(), os.path.join(save_path, 'actor.pth'))
    torch.save(critic.state_dict(), os.path.join(save_path, 'critic.pth'))

    # Also save TorchScript versions
    encoder.eval()
    rssm.eval()
    decoder.eval()
    actor.eval()

    try:
        dummy_obs = torch.randn(1, 5).to(next(encoder.parameters()).device)
        scripted_encoder = torch.jit.trace(encoder, (dummy_obs, None))
        torch.jit.save(scripted_encoder, os.path.join(save_path, 'encoder.pt'))
        print(f"  ✅ Models saved to {save_path}/")
    except Exception as e:
        print(f"  ⚠️ TorchScript export failed: {e}")

    encoder.train()
    rssm.train()
    decoder.train()
    actor.train()

def plot_training_curves(history, save_path='models/ai4i_training.png'):
    """Plot and save training curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['total_loss']) + 1)

    axes[0, 0].plot(epochs, history['total_loss'], 'b-', label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history['recon_loss'], 'g-', label='Recon Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history['reward_loss'], 'r-', label='Reward Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Reward Prediction Loss')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history['kl_loss'], 'm-', label='KL Divergence')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL')
    axes[1, 1].set_title('KL Divergence')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Training curves saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train World Model on AI4I Predictive Maintenance')
    parser.add_argument('--config', type=str, default='../configs/ai4i_predictive.json')
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--download-only', action='store_true', help='Only download dataset')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🏭 AI4I PREDICTIVE MAINTENANCE - WORLD MODEL TRAINING")
    print("="*70)

    # Step 1: Download dataset
    if not args.skip_download:
        print("\n📥 Step 1: Downloading dataset...")
        from download_datasets import download_ai4i
        success = download_ai4i()
        if not success:
            print("❌ Dataset download failed!")
            sys.exit(1)
        if args.download_only:
            print("✅ Download complete!")
            sys.exit(0)

    # Step 2: Load configuration
    print("\n📋 Step 2: Loading configuration...")
    with open(args.config, 'r') as f:
        config = json.load(f)

    for key, value in config.items():
        print(f"  {key}: {value}")

    # Step 3: Setup device
    device = setup_device(config)

    # Step 4: Initialize models
    encoder, rssm, decoder, reward_pred, actor, critic = initialize_models(config, device)

    # Step 5: Load dataset
    print("\n📦 Step 5: Loading AI4I dataset...")
    loader = AI4IDatasetLoader(config.get('data_path', 'real_world_dataset_training_with_world_models_model/data/ai4i_predictive/ai4i_2020.csv'))
    loader.load_data()

    episodes = loader.create_episodes(
        seq_len=config.get('seq_len', 50),
        episode_overlap=config.get('episode_overlap', 25)
    )

    # Step 6: Fill replay buffer
    print("\n💾 Step 6: Populating replay buffer...")
    buffer = ReplayBuffer(capacity=10000)
    for ep in tqdm(episodes, desc="Adding episodes"):
        buffer.add_episode(ep)
    print(f"  Buffer contains {len(buffer)} episodes")

    # Step 7: Training
    print("\n🚀 Step 7: Training world model...")
    num_epochs = args.epochs if args.epochs else config.get('train_world_epochs', 50)

    history = {
        'total_loss': [],
        'recon_loss': [],
        'reward_loss': [],
        'kl_loss': [],
        'eval_recon': [],
        'eval_reward': []
    }

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        losses = train_world_model_epoch(
            encoder, rssm, decoder, reward_pred,
            buffer, config, device
        )

        if losses is None:
            print(f"  Epoch {epoch+1}: Buffer too small, skipping...")
            continue

        history['total_loss'].append(losses['total_loss'])
        history['recon_loss'].append(losses['recon_loss'])
        history['reward_loss'].append(losses['reward_loss'])
        history['kl_loss'].append(losses['kl_loss'])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n  Epoch {epoch+1}/{num_epochs}:")
            print(f"    Total Loss:     {losses['total_loss']:.4f}")
            print(f"    Recon Loss:     {losses['recon_loss']:.4f}")
            print(f"    Reward Loss:    {losses['reward_loss']:.4f}")
            print(f"    KL Divergence:  {losses['kl_loss']:.4f}")

        if losses['total_loss'] < best_loss:
            best_loss = losses['total_loss']
            best_epoch = epoch + 1
            save_models(encoder, rssm, decoder, reward_pred, actor, critic, config['save_path'])

        if (epoch + 1) % 10 == 0:
            eval_metrics = evaluate_model(
                encoder, rssm, decoder, reward_pred,
                buffer, config, device
            )
            if eval_metrics:
                history['eval_recon'].append(eval_metrics['recon_error'])
                history['eval_reward'].append(eval_metrics['reward_error'])
                print(f"    Eval Recon:     {eval_metrics['recon_error']:.4f}")
                print(f"    Eval Reward:    {eval_metrics['reward_error']:.4f}")

    # Step 8: Final evaluation
    print("\n📊 Step 8: Final Evaluation...")
    final_eval = evaluate_model(
        encoder, rssm, decoder, reward_pred,
        buffer, config, device
    )

    if final_eval:
        print(f"\n  📈 Final Evaluation Metrics:")
        print(f"    Reconstruction Error: {final_eval['recon_error']:.4f}")
        print(f"    Reward Error:         {final_eval['reward_error']:.4f}")

    # Step 9: Save final models
    print("\n💾 Step 9: Saving final models...")
    save_models(encoder, rssm, decoder, reward_pred, actor, critic, config['save_path'])

    # Step 10: Plot training curves
    print("\n📊 Step 10: Generating training report...")
    plot_training_curves(history)

    # Step 11: Save training report
    report = {
        'dataset': 'AI4I Predictive Maintenance',
        'total_samples': len(loader.sensor_data),
        'episodes_created': len(episodes),
        'epochs_trained': num_epochs,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'final_metrics': final_eval if final_eval else {},
        'config': config,
        'training_history': {
            'total_loss': history['total_loss'],
            'recon_loss': history['recon_loss'],
            'reward_loss': history['reward_loss'],
            'kl_loss': history['kl_loss']
        }
    }

    with open('models/ai4i_training_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  Dataset:          AI4I Predictive Maintenance")
    print(f"  Samples:          {len(loader.sensor_data):,}")
    print(f"  Episodes:         {len(episodes):,}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Best Epoch:       {best_epoch}")
    print(f"  Best Loss:        {best_loss:.4f}")
    if final_eval:
        print(f"  Recon Error:      {final_eval['recon_error']:.4f}")
        print(f"  Reward Error:     {final_eval['reward_error']:.4f}")

    print(f"\n📁 Files created:")
    print(f"  models/encoder.pth     - Encoder weights")
    print(f"  models/rssm.pth        - RSSM weights")
    print(f"  models/decoder.pth     - Decoder weights")
    print(f"  models/reward_pred.pth - Reward predictor weights")
    print(f"  models/actor.pth       - Actor weights")
    print(f"  models/critic.pth      - Critic weights")
    print(f"  models/encoder.pt      - TorchScript encoder")
    print(f"  models/ai4i_training.png    - Training curves")
    print(f"  models/ai4i_training_report.json - Training report")

    print(f"\n🎯 Next Steps:")
    print(f"  1. Run evaluation:  python evaluate_ai4i.py")
    print(f"  2. Deploy API:      python api.py")
    print(f"  3. Benchmark:       python benchmarks/benchmark.py")

if __name__ == '__main__':
    main()