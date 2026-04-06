"""
AI4I 2020 Predictive Maintenance Dataset Loader
Loads and preprocesses industrial sensor data for world model training.
Dataset: https://archive.ics.uci.edu/dataset/601/ai4i-2020-predictive-maintenance-dataset
"""

import os
import numpy as np
import pandas as pd
from collections import deque

# Feature columns used for training (matching actual dataset column names)
SENSOR_FEATURES = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear'
]

# Failure type columns
FAILURE_COLUMNS = [
    'TWF',  # Tool Wear Failure
    'HDF',  # Heat Dissipation Failure
    'PWF',  # Power Failure
    'OSF',  # Overstrain Failure
    'RNF'   # Random Failure
]

class AI4IDatasetLoader:
    """
    Loads and preprocesses AI4I Predictive Maintenance dataset
    for training world models on industrial sensor data.
    """
    
    def __init__(self, data_path='real_world_dataset_training_with_world_models_model/data/ai4i_predictive/ai4i_2020.csv'):
        """
        Initialize loader.
        
        Args:
            data_path: Path to the CSV dataset file
        """
        self.data_path = data_path
        self.data = None
        self.sensor_data = None
        self.failure_labels = None
        self.mean = None
        self.std = None
        self.obs_dim = len(SENSOR_FEATURES)
        
    def load_data(self):
        """Load and preprocess the dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}.\n"
                f"Run: python download_datasets.py"
            )
        
        print(f"Loading AI4I dataset from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        print(f"  Raw data: {len(self.data)} samples × {len(self.data.columns)} columns")
        
        # Extract sensor features
        self.sensor_data = self.data[SENSOR_FEATURES].values.astype(np.float32)
        
        # Extract failure labels
        self.failure_labels = self.data[['Machine failure']].values.astype(np.float32).flatten()
        
        # Normalize sensor data
        self.mean = self.sensor_data.mean(axis=0)
        self.std = self.sensor_data.std(axis=0) + 1e-8
        self.sensor_data = (self.sensor_data - self.mean) / self.std
        
        print(f"  Sensor features: {self.obs_dim}")
        print(f"  Failure rate: {self.failure_labels.mean()*100:.1f}%")
        print(f"  Features normalized: mean~0, std~1")
        
        return self
    
    def compute_reward(self, idx, horizon=5):
        """
        Compute reward for a given timestep.
        
        Reward design:
        - High positive reward: Normal operation
        - High negative reward: Failure about to happen
        - Medium reward: Gradual degradation
        
        This teaches the world model to predict failures!
        """
        current_failures = self.failure_labels[idx]
        
        # Look ahead for failures
        future_idx = min(idx + horizon, len(self.failure_labels))
        future_failures = self.failure_labels[idx:future_idx]
        
        if np.any(future_failures):
            # Failure within horizon - BIG penalty
            steps_to_failure = np.argmax(future_failures)
            # Penalty increases as failure approaches
            penalty = -5.0 * (1.0 - steps_to_failure / horizon)
            return penalty
        
        elif current_failures > 0:
            # Currently failing
            return -10.0
        
        else:
            # Normal operation - positive reward
            # Higher reward for stable sensor readings
            sensor_data = self.sensor_data[idx]
            # Stability bonus for low variance
            stability = np.exp(-np.sum(np.abs(sensor_data)))
            return 1.0 + stability
    
    def create_episodes(self, seq_len=50, episode_overlap=25):
        """
        Convert time-series data into episodes for training.
        
        Args:
            seq_len: Length of each episode sequence
            episode_overlap: Overlap between consecutive episodes
            
        Returns:
            List of episodes, each episode is a list of (obs, mask, action, reward, done)
        """
        if self.sensor_data is None:
            self.load_data()
        
        episodes = []
        num_samples = len(self.sensor_data)
        
        # Create sliding window episodes
        for start in range(0, num_samples - seq_len, seq_len - episode_overlap):
            episode = []
            
            for t in range(start, start + seq_len):
                obs = self.sensor_data[t]
                
                # Action: placeholder (no control actions in this dataset)
                action = np.array([0.0])
                
                # Reward based on failure proximity
                reward = self.compute_reward(t, horizon=10)
                
                # Mask: all observations valid
                mask = np.ones(self.obs_dim, dtype=np.float32)
                
                # Done flag
                done = (t == start + seq_len - 1)
                
                episode.append((obs, mask, action, reward, done))
            
            episodes.append(episode)
        
        print(f"  Created {len(episodes)} episodes (seq_len={seq_len}, overlap={episode_overlap})")
        return episodes
    
    def get_evaluation_data(self):
        """
        Get data for evaluation/testing.
        
        Returns:
            sensor_data: Normalized sensor readings
            failure_labels: Binary failure labels
            mean: Original data mean (for denormalization)
            std: Original data std (for denormalization)
        """
        if self.sensor_data is None:
            self.load_data()
        
        return {
            'sensor_data': self.sensor_data,
            'failure_labels': self.failure_labels,
            'mean': self.mean,
            'std': self.std,
            'feature_names': SENSOR_FEATURES
        }

    def get_stats(self):
        """Get dataset statistics."""
        if self.sensor_data is None:
            self.load_data()
        
        stats = {
            'num_samples': len(self.sensor_data),
            'obs_dim': self.obs_dim,
            'failure_rate': float(self.failure_labels.mean()),
            'sensor_means': self.mean.tolist(),
            'sensor_stds': self.std.tolist()
        }
        return stats

def load_ai4i_for_training(config):
    """
    Load AI4I dataset and create episodes for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        episodes: List of episodes for replay buffer
        obs_dim: Observation dimensionality
    """
    data_path = config.get('data_path', 'real_world_dataset_training_with_world_models_model/data/ai4i_predictive/ai4i_2020.csv')
    seq_len = config.get('seq_len', 50)
    episode_overlap = config.get('episode_overlap', 25)
    
    loader = AI4IDatasetLoader(data_path)
    loader.load_data()
    
    episodes = loader.create_episodes(seq_len=seq_len, episode_overlap=episode_overlap)
    
    return episodes, loader.obs_dim, loader

if __name__ == '__main__':
    # Test the loader
    loader = AI4IDatasetLoader()
    loader.load_data()
    
    print("\n📊 Dataset Statistics:")
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n📦 Creating episodes...")
    episodes = loader.create_episodes(seq_len=50)
    
    print(f"\n✅ First episode structure:")
    if episodes:
        first_ep = episodes[0]
        print(f"  Episode length: {len(first_ep)}")
        print(f"  Obs shape: {first_ep[0][0].shape}")
        print(f"  Action shape: {first_ep[0][2].shape}")
        print(f"  Reward: {first_ep[0][3]:.4f}")
        print(f"  Done: {first_ep[0][4]}")