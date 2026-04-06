"""
World Model - Quick Start Example

This example shows how to get started with the World Model library.
Install: pip install world-model
"""

import world_model

# ============================================================
# Example 1: Training a World Model from scratch
# ============================================================

print("=" * 60)
print("Example 1: Training a World Model")
print("=" * 60)

# Create an agent with default configuration
agent = world_model.Agent()

# Train on CartPole environment
agent.train(
    env='CartPole-v1',      # Environment name
    steps=10000,            # Training steps (use 50000+ for better results)
    save_path='models/',    # Where to save models
)

# Evaluate the trained agent
avg_reward = agent.evaluate(episodes=10)
print(f"Average reward: {avg_reward}")


# ============================================================
# Example 2: Using a pre-trained model
# ============================================================

print("\n" + "=" * 60)
print("Example 2: Using a Pre-trained Model")
print("=" * 60)

# Load a pre-trained agent
agent = world_model.Agent(model_path='models/')

# Reset agent state
agent.reset()

# Run inference loop
import gymnasium as gym

env = gym.make('CartPole-v1')
obs, _ = env.reset()
total_reward = 0
done = False

while not done:
    # Get action from agent
    action = agent.step(obs, deterministic=True)
    
    # Take action in environment
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print(f"Episode reward: {total_reward}")


# ============================================================
# Example 3: Imagining future trajectories
# ============================================================

print("\n" + "=" * 60)
print("Example 3: Imagination / Planning")
print("=" * 60)

# Imagine 20 steps into the future
actions, rewards, uncertainties = agent.imagine(horizon=20)

print(f"Imagined {len(actions)} steps")
print(f"Total predicted reward: {sum(rewards):.2f}")
print(f"Average uncertainty: {sum(uncertainties)/len(uncertainties):.4f}")


# ============================================================
# Example 4: Handling missing data
# ============================================================

print("\n" + "=" * 60)
print("Example 4: Handling Missing Data")
print("=" * 60)

# Observation with some missing values
obs = [0.1, None, 0.3, 0.4]  # Second value is missing
mask = [1, 0, 1, 1]  # 1 = valid, 0 = missing

# Agent can handle missing data gracefully
action = agent.step(obs, mask=mask)
print(f"Action with missing data: {action}")


# ============================================================
# Example 5: Custom configuration
# ============================================================

print("\n" + "=" * 60)
print("Example 5: Custom Configuration")
print("=" * 60)

# Create agent with custom config
custom_config = {
    'env_name': 'Pendulum-v1',
    'obs_shape': [3],
    'action_type': 'continuous',
    'action_dim': 1,
    'latent_dim': 128,      # Larger latent space
    'hidden_dim': 512,      # Larger hidden dimension
    'total_steps': 20000,
}

agent = world_model.Agent(config=custom_config)
agent.train(
    env='Pendulum-v1',
    steps=10000,
)

print("Training complete!")