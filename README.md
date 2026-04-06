# Arynoxtech World Model - Advanced DreamerV3 with RSSM

[![PyPI version](https://img.shields.io/pypi/v/Arynoxtech_world_model.svg)](https://pypi.org/project/Arynoxtech_world_model/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A powerful World Model implementation based on **Recurrent State Space Models (RSSM)** inspired by **DreamerV3**, designed for **industrial automation, robotics, drones, autonomous vehicles**, and **intelligent systems**.

## 🆕 NEW: Cognitive Agent with Groq LLM & User Authentication

We've added a **Cognitive Agent** that combines the World Model with **Groq's LLM** to create an AI that truly thinks before it speaks!

- **User Authentication**: Secure login/register system with password hashing
- **Data Persistence**: All conversations are saved per-user and can be restored
- **Memory**: RSSM maintains conversation context in latent space
- **Imagination**: Simulates multiple response strategies before answering
- **Decision Making**: Actor-Critic selects the best response approach
- **Natural Language**: Groq LLM (llama-3.3-70b-versatile) generates human-like responses

👉 See [COGNITIVE_AGENT_GUIDE.md](COGNITIVE_AGENT_GUIDE.md) for details!

### Running the Cognitive Agent

```bash
# Set your Groq API key first
export GROQ_API_KEY="your-api-key-here"

# Run the cognitive agent
streamlit run LLM_integration/app.py
```

### Features

- **🔐 User Authentication**: Register and login with username/password
- **💾 Conversation Persistence**: All conversations are automatically saved
- **📜 Conversation History**: Load and continue previous conversations
- **📎 File Upload**: Attach text files, images, and documents for analysis
- **🧠 Cognitive Processing**: See the AI's thinking process and strategy selection

## Installation

```bash
pip install Arynoxtech_world_model
```

### Optional Dependencies

```bash
# With API support (Flask)
pip install Arynoxtech_world_model[api]

# With development tools
pip install Arynoxtech_world_model[dev]

# Everything included
pip install Arynoxtech_world_model[all]
```

## Quick Start

### Training an Agent

```python
import world_model

# Create and train an agent
agent = world_model.Agent()
agent.train(
    env='CartPole-v1',      # Any Gymnasium environment
    steps=50000,            # Training steps
    save_path='models/',    # Where to save models
)

# Evaluate
avg_reward = agent.evaluate(episodes=10)
print(f"Average reward: {avg_reward}")
```

### Using a Pre-trained Model

```python
import world_model

# Load pre-trained agent
agent = world_model.Agent(model_path='models/')

# Run inference
agent.reset()
obs = [0.1, 0.2, 0.3, 0.4]
action = agent.step(obs, deterministic=True)
print(f"Action: {action}")
```

### Imagination / Planning

```python
import world_model

agent = world_model.Agent(model_path='models/')

# Imagine future trajectories
actions, rewards, uncertainties = agent.imagine(horizon=20)

print(f"Total predicted reward: {sum(rewards):.2f}")
print(f"Average uncertainty: {sum(uncertainties)/len(uncertainties):.4f}")
```

### Handling Missing Data

```python
import world_model

agent = world_model.Agent(model_path='models/')

# Observation with missing values
obs = [0.1, None, 0.3, 0.4]  # Second value is missing
mask = [1, 0, 1, 1]           # 1 = valid, 0 = missing

action = agent.step(obs, mask=mask)
print(f"Action: {action}")
```

### Custom Configuration

```python
import world_model

# Create agent with custom config
config = {
    'env_name': 'Pendulum-v1',
    'obs_shape': [3],
    'action_type': 'continuous',
    'action_dim': 1,
    'latent_dim': 128,
    'hidden_dim': 512,
}

agent = world_model.Agent(config=config)
agent.train(env='Pendulum-v1', steps=100000)
```

## Architecture

The World Model consists of six core components:

| Component | Description |
|-----------|-------------|
| **Encoder** | Maps observations to latent embeddings |
| **RSSM** | Recurrent State Space Model for dynamics modeling |
| **Decoder** | Reconstructs observations from latent state |
| **Actor** | Policy network for action selection |
| **Critic** | Value function for state evaluation |
| **Reward Predictor** | Predicts rewards from latent state |

### Core Concepts

- **Deterministic State (h)**: GRU-based hidden state capturing temporal dependencies
- **Stochastic State (z)**: Gaussian latent variable capturing uncertainty
- **Prior**: p(z|h) - Distribution before seeing observation
- **Posterior**: q(z|h,e) - Distribution after seeing observation
- **Imagination**: Using RSSM to simulate future trajectories without environment interaction

## Features

- **Multi-Modal Support**: Vector (sensors) and Image (cameras) observations
- **Discrete & Continuous Actions**: Supports both action types
- **Robust to Missing Data**: Learnable embeddings for missing sensor values
- **Safety-Aware**: Uncertainty-based action safety checks
- **Edge Deployment**: Pruning, quantization, and TorchScript export
- **REST API**: Multi-tenant API for integration
- **Domain Randomization**: Training robustness for real-world deployment

## Trained on Industrial Datasets

| Dataset | Sensors | Samples | Training Loss | Use Case |
|---------|---------|---------|--------------|----------|
| Smart Factory IoT | 52 | 5M | 1.03 | IoT anomaly detection |
| NASA Turbofan | 24 | 5M | 1.04 | Engine health monitoring |
| Bearing Faults | 6 | 5M | 1.10 | Vibration analysis |
| AI4I Predictive | 5 | 10K | 3.44 | Multi-sensor maintenance |

**Performance**: AUC-ROC: 0.8346 | Failure Detection: 86% | False Positive Rate: 5%

## Project Structure

```
src/world_model/
├── __init__.py              # Public API exports
├── agent.py                 # Simplified high-level API
├── deployment.py            # Deployment agent
├── model/
│   ├── encoder.py           # Observation encoder
│   ├── rssm.py              # Recurrent State Space Model
│   ├── decoder.py           # Observation decoder
│   ├── actor.py             # Policy network
│   ├── critic.py            # Value network
│   └── reward_predictor.py  # Reward prediction
├── training/
│   └── trainer.py           # Training loop
└── utils/
    ├── losses.py            # Loss functions
    └── replay_buffer.py     # Experience replay
```

## API Reference

### `world_model.Agent`

```python
Agent(
    config=None,        # Dict or path to config JSON
    model_path=None,    # Path to pre-trained models
    device='auto',      # 'cpu', 'cuda', or 'auto'
)
```

**Methods:**
- `agent.step(obs, mask=None, deterministic=True)` - Get action from observation
- `agent.imagine(horizon=20, start_obs=None)` - Imagine future trajectories
- `agent.reset()` - Reset internal state
- `agent.train(env, steps, **kwargs)` - Train the agent
- `agent.save(path)` - Save models
- `agent.load(path)` - Load models
- `agent.evaluate(env_name, episodes)` - Evaluate agent

### `world_model.DreamerTrainer`

```python
DreamerTrainer(config_path='config.json')
```

**Methods:**
- `trainer.train()` - Full training loop
- `trainer.collect_experience(num_episodes)` - Collect environment data
- `trainer.train_world_model(epochs)` - Train world model
- `trainer.train_actor_critic(epochs)` - Train policy
- `trainer.evaluate(num_episodes)` - Evaluate performance

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| env_name | CartPole-v1 | Gymnasium environment |
| obs_type | vector | Observation type |
| obs_shape | [4] | Observation shape |
| action_type | discrete | Action type |
| latent_dim | 64 | Latent state dimension |
| hidden_dim | 256 | Hidden state dimension |
| batch_size | 64 | Training batch size |
| seq_len | 50 | Sequence length |
| imagine_horizon | 25 | Imagination steps |
| kl_beta | 0.1 | KL divergence weight |
| gamma | 0.99 | Discount factor |
| total_steps | 50000 | Training steps |

## Examples

See the `examples/` directory for more usage examples:

- `quick_start.py` - Basic training and inference
- Custom environment integration
- Advanced configuration options

## Developer

**Aryan Sanjay Chavan**  
📍 Chiplun, Kherdi, Maharashtra, India  
📞 +91 88579 12586  
🌐 [Portfolio](https://aryanchavanspersonalportfolio.streamlit.app)  
📧 aryaanchavan1@gmail.com  
🔗 [GitHub](https://github.com/aryaanchavan1-commits)

## License

MIT License - See [LICENSE](LICENSE) for details.

## References

1. Hafner et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*
2. Hafner et al. (2021). Mastering Atari with Discrete World Models. *ICLR 2021*
3. Hafner et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*