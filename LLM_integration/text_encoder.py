"""
Text Encoder - Converts natural language to latent space embeddings.

This module provides text encoding capabilities for the cognitive agent,
converting user messages into embeddings that can be processed by the
World Model's encoder and RSSM components.
"""

import numpy as np
import torch
from typing import List, Union, Optional
import re
import hashlib


class SimpleTextEncoder:
    """
    A lightweight text encoder that converts text to fixed-size embeddings.
    
    This uses a hash-based approach combined with character-level features
    to create consistent embeddings without requiring large pre-trained models.
    For production use, you might replace this with sentence-transformers or similar.
    """
    
    def __init__(self, embedding_dim: int = 64, vocab_size: int = 10000):
        """
        Initialize the text encoder.
        
        Args:
            embedding_dim: Dimension of the output embedding
            vocab_size: Size of the vocabulary for tokenization
        """
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Create a simple vocabulary mapping
        self._char_to_idx = {}
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789 !@#$%^&*().,?\'"-'):
            self._char_to_idx[c] = i + 1  # 0 reserved for padding
        
        # Create character embedding matrix
        self._char_embeddings = np.random.randn(
            len(self._char_to_idx) + 1, 
            embedding_dim
        ).astype(np.float32) * 0.1
        
    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        text = text.lower().strip()
        tokens = []
        
        # Simple word-level tokenization
        words = text.split()
        for word in words:
            # Hash each word to a vocabulary index
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            token_idx = word_hash % self.vocab_size
            tokens.append(token_idx)
            
        return tokens[:50]  # Limit to 50 tokens
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to a fixed-size embedding.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        tokens = self._tokenize(text)
        
        # Create embedding by averaging token embeddings
        embeddings = []
        for token in tokens:
            # Use hash-based lookup for token embedding
            token_embedding = np.random.RandomState(token).randn(
                self.embedding_dim
            ).astype(np.float32) * 0.1
            embeddings.append(token_embedding)
        
        if not embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Average pooling
        embedding = np.mean(embeddings, axis=0)
        
        # Add positional information
        position_encoding = np.zeros(self.embedding_dim, dtype=np.float32)
        for i in range(min(len(text), self.embedding_dim)):
            position_encoding[i % self.embedding_dim] += np.sin(
                i / (2 * np.pi)
            ) * 0.1
        
        embedding = embedding + position_encoding[:self.embedding_dim]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm * 2.0  # Scale to reasonable range
            
        return embedding.astype(np.float32)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return np.array([self.encode(text) for text in texts])
    
    def encode_tensor(self, text: str) -> torch.Tensor:
        """
        Encode text to a PyTorch tensor.
        
        Args:
            text: Input text string
            
        Returns:
            torch tensor of shape (1, embedding_dim)
        """
        embedding = self.encode(text)
        return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)


class ConversationEncoder:
    """
    Encodes conversation turns with role information.
    
    This encoder includes metadata about who is speaking (user vs assistant)
    and maintains conversation structure.
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize the conversation encoder.
        
        Args:
            embedding_dim: Dimension of the output embedding
        """
        self.embedding_dim = embedding_dim
        self.text_encoder = SimpleTextEncoder(embedding_dim=embedding_dim - 4)
        
    def encode_turn(self, text: str, role: str = "user") -> np.ndarray:
        """
        Encode a conversation turn with role information.
        
        Args:
            text: The message text
            role: Either "user" or "assistant"
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Encode the text
        text_embedding = self.text_encoder.encode(text)
        
        # Create role encoding (4 dimensions)
        role_encoding = np.zeros(4, dtype=np.float32)
        if role == "user":
            role_encoding[0] = 1.0
        elif role == "assistant":
            role_encoding[2] = 1.0
            
        # Combine
        full_embedding = np.concatenate([text_embedding, role_encoding])
        
        return full_embedding.astype(np.float32)


class EmotionalTextEncoder:
    """
    Text encoder that also captures emotional/sentiment information.
    
    This encoder detects basic emotional tone and includes it in the embedding.
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize the emotional text encoder.
        
        Args:
            embedding_dim: Dimension of the output embedding
        """
        self.embedding_dim = embedding_dim
        self.text_encoder = SimpleTextEncoder(embedding_dim=embedding_dim - 8)
        
        # Simple keyword-based emotion detection
        self.emotion_keywords = {
            'happy': ['happy', 'great', 'wonderful', 'excellent', 'amazing', 'love', 'joy', 'excited'],
            'sad': ['sad', 'unhappy', 'depressed', 'terrible', 'awful', 'hate', 'miserable', 'disappointed'],
            'angry': ['angry', 'furious', 'annoyed', 'irritated', 'mad', 'outraged', 'frustrated'],
            'fearful': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'panic'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
            'curious': ['curious', 'wonder', 'question', 'why', 'how', 'what', 'explain'],
            'neutral': ['the', 'a', 'is', 'are', 'was', 'were', 'have', 'has'],
        }
        
    def _detect_emotion(self, text: str) -> np.ndarray:
        """
        Detect emotional content in text.
        
        Returns:
            numpy array of shape (8,) with emotion scores
        """
        text_lower = text.lower()
        emotion_scores = np.zeros(8, dtype=np.float32)
        
        for i, (emotion, keywords) in enumerate(self.emotion_keywords.items()):
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[i] += 1.0
                    
        # Normalize
        total = np.sum(emotion_scores)
        if total > 0:
            emotion_scores = emotion_scores / total * 2.0
            
        return emotion_scores
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text with emotional information.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        text_embedding = self.text_encoder.encode(text)
        emotion_encoding = self._detect_emotion(text)
        
        return np.concatenate([text_embedding, emotion_encoding]).astype(np.float32)


# Default encoder instance
_default_encoder = None


def get_text_encoder(embedding_dim: int = 64, use_emotion: bool = False) -> SimpleTextEncoder:
    """
    Get or create a text encoder instance.
    
    Args:
        embedding_dim: Dimension of the output embedding
        use_emotion: Whether to use emotional encoding
        
    Returns:
        Text encoder instance
    """
    global _default_encoder
    
    if _default_encoder is None or _default_encoder.embedding_dim != embedding_dim:
        if use_emotion:
            _default_encoder = EmotionalTextEncoder(embedding_dim=embedding_dim)
        else:
            _default_encoder = SimpleTextEncoder(embedding_dim=embedding_dim)
            
    return _default_encoder


def encode_text(text: str, embedding_dim: int = 64) -> np.ndarray:
    """
    Convenience function to encode text.
    
    Args:
        text: Input text string
        embedding_dim: Dimension of the output embedding
        
    Returns:
        numpy array of shape (embedding_dim,)
    """
    encoder = get_text_encoder(embedding_dim=embedding_dim)
    return encoder.encode(text)


def encode_text_tensor(text: str, embedding_dim: int = 64) -> torch.Tensor:
    """
    Encode text to a PyTorch tensor.
    
    Args:
        text: Input text string
        embedding_dim: Dimension of the output embedding
        
    Returns:
        torch tensor of shape (1, embedding_dim)
    """
    encoder = get_text_encoder(embedding_dim=embedding_dim)
    return encoder.encode_tensor(text)


if __name__ == "__main__":
    # Test the encoders
    print("Testing Text Encoders...")
    
    encoder = SimpleTextEncoder(embedding_dim=64)
    
    test_texts = [
        "Hello, how are you?",
        "What is the meaning of life?",
        "The quick brown fox jumps over the lazy dog.",
        "",
        "12345 numbers and symbols!@#$%"
    ]
    
    for text in test_texts:
        embedding = encoder.encode(text)
        print(f"Text: '{text}'")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"  First 5 values: {embedding[:5]}")
        print()
    
    # Test conversation encoder
    conv_encoder = ConversationEncoder(embedding_dim=64)
    user_embedding = conv_encoder.encode_turn("Hello, how are you?", role="user")
    assistant_embedding = conv_encoder.encode_turn("I'm doing well, thank you!", role="assistant")
    
    print(f"User embedding shape: {user_embedding.shape}")
    print(f"Assistant embedding shape: {assistant_embedding.shape}")
    print(f"Embeddings are different: {not np.allclose(user_embedding, assistant_embedding)}")