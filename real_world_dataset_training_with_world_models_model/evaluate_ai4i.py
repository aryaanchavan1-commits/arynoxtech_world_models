#!/usr/bin/env python3
"""
AI4I Predictive Maintenance - World Model Evaluation & Anomaly Detection
Uses trained world model to detect anomalies and predict equipment failures.

Usage:
    python evaluate_ai4i.py                  # Full evaluation
    python evaluate_ai4i.py --plot-only      # Just generate plots
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from model.encoder import Encoder
from model.rssm import RSSM
from model.decoder import Decoder
from model.reward_predictor import RewardPredictor

# Import data loader from local folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders.ai4i_predictive import AI4IDatasetLoader

def load_trained_models(config_path='../configs/ai4i_predictive.json'):
    """Load trained world model components."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    obs_shape = config.get('obs_shape', [5])
    hidden_dim = config.get('hidden_dim', 128)
    latent_dim = config.get('latent_dim', 64)
    action_dim = config.get('action_dim', 1)
    save_path = config.get('save_path', 'models/')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(obs_type='vector', obs_shape=obs_shape, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    rssm = RSSM(action_dim=action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, is_continuous=False).to(device)
    decoder = Decoder(obs_type='vector', obs_shape=obs_shape, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    reward_pred = RewardPredictor(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    encoder.load_state_dict(torch.load(os.path.join(save_path, 'encoder.pth'), map_location=device))
    rssm.load_state_dict(torch.load(os.path.join(save_path, 'rssm.pth'), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(save_path, 'decoder.pth'), map_location=device))
    reward_pred.load_state_dict(torch.load(os.path.join(save_path, 'reward_pred.pth'), map_location=device))

    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_pred.eval()

    return encoder, rssm, decoder, reward_pred, config, device

def compute_reconstruction_errors(encoder, rssm, decoder, sensor_data, device, seq_len=50):
    """
    Compute reconstruction errors for anomaly detection.
    High reconstruction error = anomaly/potential failure.
    """
    num_samples = len(sensor_data)
    reconstruction_errors = []
    latent_states = []

    h = torch.zeros(1, rssm.hidden_dim).to(device)
    z = rssm.prior_dist(h).sample()

    with torch.no_grad():
        for t in range(num_samples):
            obs = torch.tensor(sensor_data[t:t+1], dtype=torch.float32).to(device)
            mask = torch.ones_like(obs).bool()

            obs_embed = encoder(obs, mask)
            h, z, _, _ = rssm.observe_step(
                torch.zeros(1, dtype=torch.long).to(device),
                obs_embed, h, z
            )

            pred_obs = decoder(h, z)
            recon_error = ((pred_obs - obs) ** 2).mean().item()

            reconstruction_errors.append(recon_error)
            latent_states.append(torch.cat([h, z], dim=-1).cpu().numpy())

    return np.array(reconstruction_errors), np.concatenate(latent_states, axis=0)

def detect_anomalies(reconstruction_errors, failure_labels, method='percentile', threshold_percentile=95):
    """
    Detect anomalies using reconstruction error thresholding.
    """
    if method == 'percentile':
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
    elif method == 'std':
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
    else:
        threshold = np.percentile(reconstruction_errors, threshold_percentile)

    anomalies = reconstruction_errors > threshold

    # Compute metrics if failure labels available
    if failure_labels is not None and len(failure_labels) > 0:
        # AUC-ROC
        try:
            auc = roc_auc_score(failure_labels, reconstruction_errors)
        except:
            auc = 0.0

        # F1 Score
        preds = anomalies.astype(int)
        f1 = f1_score(failure_labels[:len(preds)], preds[:len(failure_labels)], zero_division=0)

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(
            failure_labels[:len(reconstruction_errors)],
            reconstruction_errors[:len(failure_labels)]
        )

        return {
            'anomalies': anomalies,
            'threshold': threshold,
            'auc_roc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

    return {'anomalies': anomalies, 'threshold': threshold}

def plot_evaluation_results(sensor_data, failure_labels, reconstruction_errors, anomaly_results, feature_names, save_dir='models/evaluation'):
    """Generate comprehensive evaluation plots."""
    os.makedirs(save_dir, exist_ok=True)

    num_features = sensor_data.shape[1]
    num_samples = len(sensor_data)

    # Plot 1: Sensor Readings Over Time with Anomaly Highlights
    fig, axes = plt.subplots(num_features + 1, 1, figsize=(16, 3 * (num_features + 1)))

    for i in range(num_features):
        axes[i].plot(range(num_samples), sensor_data[:, i], 'b-', alpha=0.7, label=feature_names[i])
        anomaly_idx = np.where(anomaly_results['anomalies'][:num_samples])[0]
        if len(anomaly_idx) > 0:
            axes[i].scatter(anomaly_idx, sensor_data[anomaly_idx, i], color='red', s=20, label='Anomaly', zorder=5)
        axes[i].set_ylabel(feature_names[i])
        axes[i].grid(True)
        axes[i].legend()

    # Plot failure labels if available
    if failure_labels is not None:
        axes[-1].fill_between(range(num_samples), failure_labels[:num_samples], alpha=0.3, color='red', label='Actual Failure')
        axes[-1].set_ylabel('Failure Label')
        axes[-1].set_ylim(-0.1, 1.1)
        axes[-1].legend()

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensor_readings_with_anomalies.png'), dpi=150)
    plt.close()

    # Plot 2: Reconstruction Error Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(reconstruction_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=anomaly_results['threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold: {anomaly_results["threshold"]:.4f}')
    axes[0].set_xlabel('Reconstruction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Reconstruction Errors')
    axes[0].legend()
    axes[0].grid(True)

    # Plot anomaly scores over time
    axes[1].plot(range(num_samples), reconstruction_errors[:num_samples], 'b-', alpha=0.7)
    axes[1].axhline(y=anomaly_results['threshold'], color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Reconstruction Error')
    axes[1].set_title('Anomaly Scores Over Time')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_error_analysis.png'), dpi=150)
    plt.close()

    # Plot 3: Precision-Recall Curve (if failure labels available)
    if 'precision' in anomaly_results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(anomaly_results['recall'], anomaly_results['precision'], 'b-', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (AUC: {anomaly_results["auc_roc"]:.4f})')
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=150)
        plt.close()

    print(f"  📊 Evaluation plots saved to {save_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Evaluate World Model on AI4I Predictive Maintenance')
    parser.add_argument('--config', type=str, default='../configs/ai4i_predictive.json')
    parser.add_argument('--threshold', type=str, default='percentile', choices=['percentile', 'std'])
    parser.add_argument('--percentile', type=int, default=95)
    parser.add_argument('--plot-only', action='store_true', help='Skip evaluation, just generate plots')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🔍 AI4I PREDICTIVE MAINTENANCE - WORLD MODEL EVALUATION")
    print("="*70)

    # Load trained models
    print("\n📦 Loading trained models...")
    encoder, rssm, decoder, reward_pred, config, device = load_trained_models(args.config)
    print(f"  Device: {device}")

    # Load dataset
    print("\n📊 Loading AI4I dataset...")
    loader = AI4IDatasetLoader(config.get('data_path', 'real_world_dataset_training_with_world_models_model/data/ai4i_predictive/ai4i_2020.csv'))
    loader.load_data()

    eval_data = loader.get_evaluation_data()
    sensor_data = eval_data['sensor_data']
    failure_labels = eval_data['failure_labels']
    feature_names = eval_data['feature_names']

    print(f"  Samples: {len(sensor_data)}")
    print(f"  Features: {feature_names}")
    print(f"  Failure rate: {failure_labels.mean()*100:.1f}%")

    # Compute reconstruction errors
    print("\n🔎 Computing reconstruction errors...")
    recon_errors, latent_states = compute_reconstruction_errors(
        encoder, rssm, decoder, sensor_data, device, seq_len=config.get('seq_len', 50)
    )
    print(f"  Mean recon error: {recon_errors.mean():.4f}")
    print(f"  Std recon error: {recon_errors.std():.4f}")
    print(f"  Min/Max: {recon_errors.min():.4f} / {recon_errors.max():.4f}")

    # Detect anomalies
    print(f"\n🚨 Detecting anomalies (method: {args.threshold}, percentile: {args.percentile})...")
    anomaly_results = detect_anomalies(
        recon_errors, failure_labels,
        method=args.threshold, threshold_percentile=args.percentile
    )

    num_anomalies = anomaly_results['anomalies'].sum()
    anomaly_rate = num_anomalies / len(recon_errors) * 100

    print(f"  Threshold: {anomaly_results['threshold']:.4f}")
    print(f"  Anomalies detected: {num_anomalies} ({anomaly_rate:.1f}%)")

    if 'auc_roc' in anomaly_results:
        print(f"  AUC-ROC: {anomaly_results['auc_roc']:.4f}")
        print(f"  F1 Score: {anomaly_results['f1_score']:.4f}")

    # Generate evaluation plots
    print("\n📊 Generating evaluation plots...")
    plot_evaluation_results(
        sensor_data, failure_labels, recon_errors, anomaly_results,
        feature_names, save_dir='models/evaluation'
    )

    # Save evaluation report
    report = {
        'dataset': 'AI4I Predictive Maintenance',
        'total_samples': int(len(sensor_data)),
        'failure_samples': int(failure_labels.sum()),
        'failure_rate': float(failure_labels.mean()),
        'reconstruction_errors': {
            'mean': float(recon_errors.mean()),
            'std': float(recon_errors.std()),
            'min': float(recon_errors.min()),
            'max': float(recon_errors.max()),
            'median': float(np.median(recon_errors))
        },
        'anomaly_detection': {
            'method': args.threshold,
            'threshold': float(anomaly_results['threshold']),
            'num_anomalies': int(num_anomalies),
            'anomaly_rate_percent': float(anomaly_rate)
        }
    }

    if 'auc_roc' in anomaly_results:
        report['anomaly_detection']['auc_roc'] = float(anomaly_results['auc_roc'])
        report['anomaly_detection']['f1_score'] = float(anomaly_results['f1_score'])

    with open('models/evaluation/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  Dataset:               AI4I Predictive Maintenance")
    print(f"  Total Samples:         {len(sensor_data):,}")
    print(f"  Actual Failures:       {int(failure_labels.sum()):,}")
    print(f"  Detected Anomalies:    {num_anomalies:,}")
    print(f"  Mean Recon Error:      {recon_errors.mean():.4f}")
    print(f"  Anomaly Threshold:     {anomaly_results['threshold']:.4f}")
    if 'auc_roc' in anomaly_results:
        print(f"  AUC-ROC:               {anomaly_results['auc_roc']:.4f}")
        print(f"  F1 Score:              {anomaly_results['f1_score']:.4f}")

    print(f"\n📁 Files created:")
    print(f"  models/evaluation/sensor_readings_with_anomalies.png")
    print(f"  models/evaluation/reconstruction_error_analysis.png")
    print(f"  models/evaluation/precision_recall_curve.png")
    print(f"  models/evaluation/evaluation_report.json")

    print(f"\n🎯 Next Steps:")
    print(f"  1. Deploy API:  python api.py")
    print(f"  2. Benchmark:   python benchmarks/benchmark.py")

if __name__ == '__main__':
    main()