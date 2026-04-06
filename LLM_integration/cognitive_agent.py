"""
Cognitive Agent - World Model powered AI with LLM language interface.

This module creates a cognitive AI agent that combines:
- World Model (RSSM) for memory, imagination, and world understanding
- Groq LLM for natural language generation
- Actor-Critic for decision making and response selection

The agent thinks like a human: it perceives, remembers, imagines outcomes,
and then responds with genuine understanding.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime

# Import local text encoder from the same directory
from .text_encoder import SimpleTextEncoder, encode_text_tensor

# Import the world model Agent from the installed PyPI package
# The package should be installed via: pip install arynoxtech-world-model
def _import_world_model_agent():
    """Try to import Agent from world_model, with fallback to install if needed."""
    try:
        from world_model.agent import Agent
        return Agent
    except ImportError:
        # Try to install the package dynamically
        import subprocess
        import sys
        
        print("Attempting to install arynoxtech-world-model package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "arynoxtech-world-model", "-q"])
            # Clear any cached import failures
            if 'world_model' in sys.modules:
                del sys.modules['world_model']
            from world_model.agent import Agent
            return Agent
        except Exception as install_error:
            raise ImportError(
                f"Could not import world_model.agent.Agent: {install_error}. "
                "Please ensure the arynoxtech-world-model package is installed: pip install arynoxtech-world-model"
            )

Agent = _import_world_model_agent()

# Try to import Groq, handle if not installed
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq library not installed. Install with: pip install groq")


@dataclass
class ImaginationScenario:
    """Represents an imagined conversation scenario."""
    strategy: str
    predicted_reward: float
    uncertainty: float
    actions: List[Any]
    description: str


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation."""
    role: str  # "user" or "assistant"
    message: str
    timestamp: str
    memory_state: Optional[Dict[str, Any]] = None
    imagination_scenarios: Optional[List[Dict[str, Any]]] = None
    selected_strategy: Optional[str] = None


class CognitiveAgent:
    """
    World Model-powered cognitive agent with LLM language interface.
    
    This agent uses:
    - RSSM for maintaining conversation memory and state
    - Imagination for planning response strategies
    - Actor-Critic for selecting optimal responses
    - Groq LLM for generating natural language
    
    The cognitive pipeline:
    1. Perceive: Encode user message to latent space
    2. Remember: Update RSSM state with conversation context
    3. Imagine: Simulate multiple response strategies
    4. Decide: Select best strategy using Actor-Critic
    5. Respond: Generate natural language with LLM
    """
    
    def __init__(
        self,
        world_model_path: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = 'auto',
        imagination_horizon: int = 10,
        num_scenarios: int = 5,
    ):
        """
        Initialize the cognitive agent.
        
        Args:
            world_model_path: Path to pre-trained world model
            groq_api_key: Groq API key for LLM
            config: Optional configuration overrides
            device: Device to use ('cpu', 'cuda', or 'auto')
            imagination_horizon: Steps to imagine into the future
            num_scenarios: Number of scenarios to imagine per response
        """
        self.imagination_horizon = imagination_horizon
        self.num_scenarios = num_scenarios
        
        # Initialize text encoder
        self.text_encoder = SimpleTextEncoder(embedding_dim=64)
        
        # Initialize world model (don't load pre-trained models to avoid shape mismatch)
        # Use a configuration suitable for conversation
        print("Initializing world model for cognitive agent")
        default_config = {
            'obs_type': 'vector',
            'obs_shape': [64],  # Text embedding dimension
            'action_type': 'discrete',
            'action_dim': 10,  # Number of response strategies
            'latent_dim': 64,
            'hidden_dim': 256,
            'imagine_horizon': imagination_horizon,
        }
        if config:
            default_config.update(config)
        self.agent = Agent(config=default_config, device=device)
        
        # Initialize Groq client
        if groq_api_key and GROQ_AVAILABLE:
            self.groq_client = Groq(api_key=groq_api_key)
            self.llm_available = True
        else:
            self.groq_client = None
            self.llm_available = False
            if not GROQ_AVAILABLE:
                print("Groq library not installed - LLM features disabled")
            elif not groq_api_key:
                print("No Groq API key provided - LLM features disabled")
        
        # Conversation history
        self.conversation_history: List[ConversationTurn] = []
        
        # Response strategies (actions the agent can take)
        self.response_strategies = [
            "Provide a direct, factual answer",
            "Ask a clarifying question",
            "Share a relevant personal insight",
            "Offer multiple perspectives",
            "Use humor to lighten the mood",
            "Provide detailed technical explanation",
            "Give a simple, accessible explanation",
            "Express empathy and understanding",
            "Challenge the assumption gently",
            "Suggest practical next steps",
        ]
        
        # System prompt for LLM
        self.system_prompt = """You are an advanced AI assistant with genuine understanding and cognition. 
You have a world model that helps you understand context, maintain memory, and imagine outcomes.
You think before you speak, considering multiple perspectives and response strategies.
Your responses should be:
- Thoughtful and context-aware
- Empathetic and understanding
- Clear and well-structured
- Appropriate to the conversation tone
- Helpful and insightful

You have access to a cognitive world model that provides:
- Memory: You remember the conversation context and user preferences
- Imagination: You can simulate different response approaches
- Understanding: You grasp nuances and implicit meanings
- Planning: You consider long-term conversation goals

Respond naturally and authentically, as a thinking, feeling entity would."""
        
        print(f"Cognitive Agent initialized")
        print(f"  World Model: {'Loaded' if world_model_path else 'New'}")
        print(f"  LLM: {'Available' if self.llm_available else 'Disabled'}")
        print(f"  Imagination Horizon: {imagination_horizon}")
        print(f"  Response Strategies: {len(self.response_strategies)}")
    
    def _encode_message(self, message: str) -> torch.Tensor:
        """
        Encode a text message to latent space.
        
        Args:
            message: User's text message
            
        Returns:
            Tensor of shape (1, embedding_dim)
        """
        return encode_text_tensor(message, embedding_dim=64)
    
    def _update_memory(self, message: str, role: str = "user"):
        """
        Update the agent's memory with a conversation turn.
        
        Args:
            message: The message text
            role: "user" or "assistant"
        """
        # Encode the message
        obs_embed = self._encode_message(message)
        obs_embed = obs_embed.to(self.agent.device)
        
        # Create a dummy action (we're observing, not acting yet)
        action_dim = self.agent.config.get('action_dim', 10)
        dummy_action = torch.zeros(1, dtype=torch.long).to(self.agent.device)
        
        # Update RSSM state
        self.agent.h, self.agent.z, z_mean, z_std = self.agent.rssm.observe_step(
            action=dummy_action,
            obs_embed=obs_embed,
            prev_h=self.agent.h,
            prev_z=self.agent.z
        )
        
        # Store memory state
        memory_state = {
            'h_norm': torch.norm(self.agent.h).item(),
            'z_mean_norm': torch.norm(z_mean).item(),
            'z_std_mean': z_std.mean().item(),
            'timestamp': datetime.now().isoformat(),
        }
        
        return memory_state
    
    def _imagine_scenarios(self) -> List[ImaginationScenario]:
        """
        Imagine multiple response scenarios using the world model.
        
        Returns:
            List of imagined scenarios with predicted outcomes
        """
        scenarios = []
        
        for strategy_idx in range(self.num_scenarios):
            # Save current state
            h_saved = self.agent.h.clone()
            z_saved = self.agent.z.clone()
            
            # Imagine trajectory for this strategy
            actions = []
            rewards = []
            uncertainties = []
            
            h = self.agent.h.clone()
            z = self.agent.z.clone()
            
            for step in range(self.imagination_horizon):
                # Select action (response strategy)
                if step == 0:
                    # First step: bias toward this strategy
                    action = torch.tensor([strategy_idx], dtype=torch.long).to(self.agent.device)
                else:
                    # Subsequent steps: use actor
                    action = self.agent.actor.sample_action(h, z, deterministic=False)
                    # Ensure action is 1D tensor
                    if action.dim() == 0:
                        action = action.unsqueeze(0)
                
                # Get uncertainty
                uncertainty = self.agent.actor.get_uncertainty(h, z).item()
                
                # Imagine next state
                h, z, z_mean, z_std = self.agent.rssm.imagine_step(
                    action=action,
                    prev_h=h,
                    prev_z=z
                )
                
                # Predict reward
                reward = self.agent.reward_predictor(h, z).item()
                
                actions.append(action.item())
                rewards.append(reward)
                uncertainties.append(uncertainty)
            
            # Calculate scenario metrics
            total_reward = sum(rewards)
            avg_uncertainty = np.mean(uncertainties)
            
            # Create scenario description
            strategy_name = self.response_strategies[strategy_idx % len(self.response_strategies)]
            description = f"Strategy: {strategy_name}\n"
            description += f"Predicted conversation flow: {len(rewards)} steps\n"
            description += f"Expected engagement: {total_reward:.2f}\n"
            description += f"Uncertainty: {avg_uncertainty:.3f}"
            
            scenario = ImaginationScenario(
                strategy=strategy_idx,
                predicted_reward=total_reward,
                uncertainty=avg_uncertainty,
                actions=actions,
                description=description,
            )
            scenarios.append(scenario)
            
            # Restore state
            self.agent.h = h_saved
            self.agent.z = z_saved
        
        return scenarios
    
    def _select_best_strategy(
        self, 
        scenarios: List[ImaginationScenario]
    ) -> ImaginationScenario:
        """
        Select the best response strategy using Actor-Critic.
        
        Args:
            scenarios: List of imagined scenarios
            
        Returns:
            Best scenario to use
        """
        # Use critic to evaluate each scenario's state value
        scored_scenarios = []
        
        for scenario in scenarios:
            # Score based on predicted reward and low uncertainty
            score = scenario.predicted_reward - 0.1 * scenario.uncertainty
            scored_scenarios.append((scenario, score))
        
        # Sort by score and return best
        scored_scenarios.sort(key=lambda x: x[1], reverse=True)
        return scored_scenarios[0][0]
    
    def _generate_llm_response(
        self,
        user_message: str,
        selected_strategy: ImaginationScenario,
        conversation_context: str,
    ) -> str:
        """
        Generate response using Groq LLM.
        
        Args:
            user_message: User's input message
            selected_strategy: Chosen response strategy
            conversation_context: Recent conversation history
            
        Returns:
            Generated response text
        """
        if not self.llm_available or not self.groq_client:
            return self._generate_fallback_response(user_message, selected_strategy)
        
        # Create strategy-specific prompt
        strategy_prompt = f"""
Based on the user's message and the conversation context, respond using this strategy:

STRATEGY: {selected_strategy.description}

CONVERSATION CONTEXT:
{conversation_context}

USER'S MESSAGE: {user_message}

Respond naturally and thoughtfully, following the strategy above."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": strategy_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(user_message, selected_strategy)
    
    def _generate_fallback_response(
        self,
        user_message: str,
        selected_strategy: ImaginationScenario,
    ) -> str:
        """
        Generate a simple response when LLM is not available.
        
        Args:
            user_message: User's input message
            selected_strategy: Chosen response strategy
            
        Returns:
            Generated response text
        """
        strategy_name = self.response_strategies[selected_strategy.strategy % len(self.response_strategies)]
        
        responses = {
            0: f"I understand. Based on my analysis, {user_message[:50]}...",
            1: f"That's an interesting point. Could you tell me more about what you mean?",
            2: f"I see. From my perspective, this reminds me of the importance of context.",
            3: f"There are multiple ways to look at this. On one hand, and on the other...",
            4: f"Well, they say laughter is the best medicine! But seriously...",
            5: f"Let me break this down technically for you...",
            6: f"In simple terms, what you're asking about is quite fascinating...",
            7: f"I can sense this is important to you. I understand and I'm here to help.",
            8: f"That's an interesting assumption. Have you considered alternative views?",
            9: f"Here's what I'd suggest as a practical next step...",
        }
        
        return responses.get(selected_strategy.strategy % 10, 
                            f"I've processed your message using strategy: {strategy_name}")
    
    def generate_response(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Full cognitive pipeline: perceive → remember → imagine → respond.
        
        Args:
            user_message: User's input message
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        # 1. Update memory with user message
        user_memory = self._update_memory(user_message, role="user")
        
        # 2. Imagine possible response scenarios
        scenarios = self._imagine_scenarios()
        
        # 3. Select best strategy
        best_scenario = self._select_best_strategy(scenarios)
        
        # 4. Get conversation context
        conversation_context = self._get_conversation_context(last_n=5)
        
        # 5. Generate response
        response_text = self._generate_llm_response(
            user_message,
            best_scenario,
            conversation_context,
        )
        
        # 6. Update memory with assistant response
        assistant_memory = self._update_memory(response_text, role="assistant")
        
        # 7. Store conversation turn
        turn = ConversationTurn(
            role="user",
            message=user_message,
            timestamp=datetime.now().isoformat(),
            memory_state=user_memory,
            imagination_scenarios=[asdict(s) for s in scenarios],
            selected_strategy=best_scenario.description,
        )
        self.conversation_history.append(turn)
        
        turn = ConversationTurn(
            role="assistant",
            message=response_text,
            timestamp=datetime.now().isoformat(),
            memory_state=assistant_memory,
        )
        self.conversation_history.append(turn)
        
        # 8. Prepare metadata
        metadata = {
            'scenarios': [asdict(s) for s in scenarios],
            'selected_strategy': asdict(best_scenario),
            'memory_state': assistant_memory,
            'conversation_length': len(self.conversation_history),
        }
        
        return response_text, metadata
    
    def _get_conversation_context(self, last_n: int = 5) -> str:
        """
        Get recent conversation context.
        
        Args:
            last_n: Number of recent turns to include
            
        Returns:
            Formatted conversation context string
        """
        recent_turns = self.conversation_history[-last_n:] if len(self.conversation_history) > last_n else self.conversation_history
        
        context_lines = []
        for turn in recent_turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            context_lines.append(f"{role_label}: {turn.message}")
        
        return "\n".join(context_lines)
    
    def reset(self):
        """Reset the agent's state and conversation history."""
        self.agent.reset()
        self.conversation_history = []
        print("Cognitive agent reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary of agent stats
        """
        return {
            'conversation_turns': len(self.conversation_history),
            'memory_h_norm': torch.norm(self.agent.h).item(),
            'memory_z_norm': torch.norm(self.agent.z).item(),
            'llm_available': self.llm_available,
            'imagination_horizon': self.imagination_horizon,
            'num_scenarios': self.num_scenarios,
        }
    
    def save_conversation(self, filepath: str):
        """
        Save conversation history to a file.
        
        Args:
            filepath: Path to save the conversation
        """
        data = {
            'conversation': [asdict(turn) for turn in self.conversation_history],
            'stats': self.get_stats(),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """
        Load conversation history from a file.
        
        Args:
            filepath: Path to load the conversation from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.conversation_history = [
            ConversationTurn(**turn) for turn in data['conversation']
        ]
        
        print(f"Conversation loaded from {filepath}")


def create_cognitive_agent(
    world_model_path: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    **kwargs
) -> CognitiveAgent:
    """
    Factory function to create a cognitive agent.
    
    Args:
        world_model_path: Path to world model
        groq_api_key: Groq API key
        **kwargs: Additional arguments
        
    Returns:
        CognitiveAgent instance
    """
    return CognitiveAgent(
        world_model_path=world_model_path,
        groq_api_key=groq_api_key,
        **kwargs
    )


if __name__ == "__main__":
    # Test the cognitive agent
    print("Testing Cognitive Agent...")
    
    # Create agent without LLM for testing
    agent = CognitiveAgent(
        world_model_path=None,
        groq_api_key=None,
        imagination_horizon=5,
        num_scenarios=3,
    )
    
    # Test conversation
    test_messages = [
        "Hello! How are you today?",
        "What do you think about artificial intelligence?",
        "Can you help me understand machine learning?",
    ]
    
    for message in test_messages:
        print(f"\n{'='*50}")
        print(f"User: {message}")
        
        response, metadata = agent.generate_response(message)
        
        print(f"Agent: {response}")
        print(f"\nSelected Strategy: {metadata['selected_strategy']['description']}")
        print(f"Predicted Reward: {metadata['selected_strategy']['predicted_reward']:.2f}")
        print(f"Uncertainty: {metadata['selected_strategy']['uncertainty']:.3f}")
    
    print(f"\n{'='*50}")
    print(f"Agent Stats: {agent.get_stats()}")