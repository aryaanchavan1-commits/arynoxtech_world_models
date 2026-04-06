"""
Performance benchmarks for World Model.
Compare latency, memory, accuracy vs alternatives.
"""

import time
import torch
import numpy as np
from deployment import WorldModelAgent

def benchmark_inference(agent, obs_samples=100):
    """Benchmark inference latency."""
    times = []
    for _ in range(obs_samples):
        obs = np.random.randn(4)  # Example obs
        start = time.time()
        action = agent.step(obs)
        end = time.time()
        times.append(end - start)

    avg_latency = np.mean(times) * 1000  # ms
    p95_latency = np.percentile(times, 95) * 1000
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    return avg_latency, p95_latency

def benchmark_memory(agent):
    """Benchmark memory usage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        obs = np.random.randn(4)
        agent.step(obs)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Peak GPU Memory: {memory_mb:.2f} MB")
        return memory_mb
    else:
        print("CPU memory benchmark not implemented")
        return 0

def benchmark_accuracy(agent, env_name='CartPole-v1', episodes=10):
    """Benchmark task performance."""
    import gymnasium as gym
    env = gym.make(env_name)
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.step(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_reward += ep_reward
    avg_reward = total_reward / episodes
    print(f"Average Episode Reward: {avg_reward:.2f}")
    env.close()
    return avg_reward

def compare_alternatives():
    """Compare vs simple RL baselines."""
    print("Comparison with Random Policy:")
    # Random policy benchmark
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    random_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        reward = 0
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward += r
        random_rewards.append(reward)
    print(f"Random Policy Avg Reward: {np.mean(random_rewards):.2f}")

if __name__ == '__main__':
    agent = WorldModelAgent()
    print("Benchmarking World Model...")
    benchmark_inference(agent)
    benchmark_memory(agent)
    benchmark_accuracy(agent)
    compare_alternatives()