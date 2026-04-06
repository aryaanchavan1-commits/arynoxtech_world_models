import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Simple replay buffer for storing episodes.
    Each episode is a list of (obs, mask, action, reward, done)
    """
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, episode):
        """
        Add an episode: list of (obs, mask, action, reward, done)
        """
        self.buffer.append(episode)

    def sample_batch(self, batch_size, seq_len):
        """
        Sample batch of sequences.
        Returns:
            obs_seq: (batch, seq_len, *obs_shape)
            mask_seq: (batch, seq_len, *obs_shape) or None
            action_seq: (batch, seq_len, action_dim)
            reward_seq: (batch, seq_len)
            done_seq: (batch, seq_len)
        """
        episodes = []
        while len(episodes) < batch_size:
            ep = random.choice(self.buffer)
            if len(ep) >= seq_len:
                episodes.append(ep)

        obs_seqs = []
        mask_seqs = []
        action_seqs = []
        reward_seqs = []
        done_seqs = []

        for ep in episodes:
            # Take random subsequence of length seq_len
            start = random.randint(0, len(ep) - seq_len)
            seq = ep[start:start + seq_len]

            obs = np.array([s[0] for s in seq])
            mask = np.array([s[1] for s in seq])
            action = np.array([s[2] for s in seq])
            reward = np.array([s[3] for s in seq])
            done = np.array([s[4] for s in seq])

            obs_seqs.append(obs)
            mask_seqs.append(mask)
            action_seqs.append(action)
            reward_seqs.append(reward)
            done_seqs.append(done)

        return (
            np.array(obs_seqs),
            np.array(mask_seqs),
            np.array(action_seqs),
            np.array(reward_seqs),
            np.array(done_seqs),
        )

    def __len__(self):
        return len(self.buffer)