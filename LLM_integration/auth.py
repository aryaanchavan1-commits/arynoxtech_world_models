"""
Enterprise-Grade User Authentication Module for Arynoxtech Cognitive Agent.

This module provides:
- User registration with bcrypt password hashing
- User login/logout functionality
- Session management
- Per-user data persistence (conversation history, preferences)
- Rate limiting for security
- Password strength validation
"""

import bcrypt
import json
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re


# Data directory for user storage
DATA_DIR = Path("user_data")
USERS_FILE = DATA_DIR / "users.json"

# Rate limiting configuration
RATE_LIMIT_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 300  # 5 minutes in seconds

# Password requirements
MIN_PASSWORD_LENGTH = 6
MAX_PASSWORD_LENGTH = 128
PASSWORD_COMPLEXITY = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{6,128}$"
)


@dataclass
class User:
    """User data structure."""
    username: str
    password_hash: str
    created_at: str
    last_login: Optional[str] = None
    total_conversations: int = 0
    total_messages: int = 0
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[str] = None


class RateLimiter:
    """Simple in-memory rate limiter for login attempts."""
    
    def __init__(self, max_attempts: int = RATE_LIMIT_ATTEMPTS, 
                 window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: Dict[str, List[float]] = defaultdict(list)
    
    def is_rate_limited(self, identifier: str) -> bool:
        """Check if an identifier is rate limited."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old attempts
        self._attempts[identifier] = [
            t for t in self._attempts[identifier] if t > window_start
        ]
        
        return len(self._attempts[identifier]) >= self.max_attempts
    
    def record_attempt(self, identifier: str):
        """Record a failed attempt."""
        self._attempts[identifier].append(time.time())
    
    def reset(self, identifier: str):
        """Reset attempts for an identifier (on successful login)."""
        self._attempts[identifier] = []


# Global rate limiter
rate_limiter = RateLimiter()


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password strength.
    
    Requirements:
    - At least 6 characters
    - At most 128 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    
    Returns:
        Tuple of (is_valid, message)
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"
    
    if len(password) > MAX_PASSWORD_LENGTH:
        return False, f"Password must be at most {MAX_PASSWORD_LENGTH} characters long"
    
    if not PASSWORD_COMPLEXITY.match(password):
        return False, (
            "Password must contain at least one uppercase letter, one lowercase letter, "
            "one digit, and one special character (@$!%*?&)"
        )
    
    return True, "Password meets requirements"


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate username.
    
    Requirements:
    - At least 3 characters
    - Alphanumeric, underscores, and hyphens only
    
    Returns:
        Tuple of (is_valid, message)
    """
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens"
    
    return True, "Username is valid"


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        Bcrypt hash string
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a password against stored bcrypt hash.
    
    Args:
        password: Plain text password to verify
        stored_hash: Stored bcrypt hash
    
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            stored_hash.encode('utf-8')
        )
    except (ValueError, TypeError):
        return False


def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)


def load_users() -> Dict[str, Dict[str, Any]]:
    """Load all users from the users file."""
    ensure_data_dir()
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_users(users: Dict[str, Dict[str, Any]]):
    """Save users to the users file."""
    ensure_data_dir()
    # Write to temp file first, then rename for atomicity
    temp_file = USERS_FILE.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(users, f, indent=2)
    temp_file.replace(USERS_FILE)


def register_user(username: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user.
    
    Args:
        username: Desired username
        password: Plain text password
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Validate username
    is_valid, message = validate_username(username)
    if not is_valid:
        return False, message
    
    # Validate password
    is_valid, message = validate_password(password)
    if not is_valid:
        return False, message
    
    ensure_data_dir()
    users = load_users()
    
    if username.lower() in users:
        return False, "Username already exists"
    
    # Hash password and create user
    password_hash = hash_password(password)
    
    user = User(
        username=username,
        password_hash=password_hash,
        created_at=datetime.now().isoformat(),
    )
    
    users[username.lower()] = asdict(user)
    save_users(users)
    
    # Create user's data directory
    user_dir = DATA_DIR / username.lower()
    user_dir.mkdir(exist_ok=True)
    (user_dir / "conversations").mkdir(exist_ok=True)
    
    return True, f"User '{username}' registered successfully"


def login_user(username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticate a user with rate limiting.
    
    Args:
        username: Username
        password: Plain text password
    
    Returns:
        Tuple of (success: bool, message: str, user_data: Optional[Dict])
    """
    if not username or not password:
        return False, "Username and password are required", None
    
    # Check rate limiting
    if rate_limiter.is_rate_limited(username.lower()):
        return False, "Too many failed attempts. Please try again later.", None
    
    users = load_users()
    user_data = users.get(username.lower())
    
    if not user_data:
        rate_limiter.record_attempt(username.lower())
        return False, "Invalid username or password", None
    
    # Check if account is locked
    if user_data.get('locked_until'):
        locked_until = datetime.fromisoformat(user_data['locked_until'])
        if datetime.now() < locked_until:
            return False, "Account is temporarily locked. Please try again later.", None
        else:
            # Unlock account
            user_data['locked_until'] = None
            user_data['failed_attempts'] = 0
            users[username.lower()] = user_data
            save_users(users)
    
    if not verify_password(password, user_data['password_hash']):
        # Record failed attempt
        rate_limiter.record_attempt(username.lower())
        user_data['failed_attempts'] = user_data.get('failed_attempts', 0) + 1
        
        # Lock account after too many failures
        if user_data['failed_attempts'] >= RATE_LIMIT_ATTEMPTS:
            lock_duration = timedelta(minutes=15)
            user_data['locked_until'] = (datetime.now() + lock_duration).isoformat()
            save_users(users)
            return False, "Account locked due to too many failed attempts.", None
        
        save_users(users)
        return False, "Invalid username or password", None
    
    # Successful login - reset attempts
    rate_limiter.reset(username.lower())
    user_data['failed_attempts'] = 0
    user_data['locked_until'] = None
    user_data['last_login'] = datetime.now().isoformat()
    users[username.lower()] = user_data
    save_users(users)
    
    return True, f"Welcome back, {username}!", user_data


def get_user_data(username: str) -> Optional[Dict[str, Any]]:
    """Get user data."""
    users = load_users()
    return users.get(username.lower())


def user_exists(username: str) -> bool:
    """Check if a user exists."""
    users = load_users()
    return username.lower() in users


def get_user_conversations_dir(username: str) -> Path:
    """Get the conversations directory for a user."""
    return DATA_DIR / username.lower() / "conversations"


def save_user_conversation(username: str, conversation_id: str, data: Dict[str, Any]):
    """
    Save a conversation for a user.
    
    Args:
        username: Username
        conversation_id: Unique conversation identifier
        data: Conversation data to save
    """
    conv_dir = get_user_conversations_dir(username)
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = conv_dir / f"{conversation_id}.json"
    temp_file = filepath.with_suffix('.tmp')
    
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    temp_file.replace(filepath)


def load_user_conversation(username: str, conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation for a user.
    
    Args:
        username: Username
        conversation_id: Unique conversation identifier
    
    Returns:
        Conversation data or None if not found
    """
    filepath = get_user_conversations_dir(username) / f"{conversation_id}.json"
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def list_user_conversations(username: str) -> List[Dict[str, Any]]:
    """
    List all conversations for a user.
    
    Args:
        username: Username
    
    Returns:
        List of conversation metadata
    """
    conv_dir = get_user_conversations_dir(username)
    
    if not conv_dir.exists():
        return []
    
    conversations = []
    for filepath in sorted(conv_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            conversations.append({
                'id': filepath.stem,
                'timestamp': data.get('timestamp', 'Unknown'),
                'message_count': data.get('stats', {}).get('conversation_turns', 0),
                'filepath': str(filepath),
            })
        except (json.JSONDecodeError, Exception):
            continue
    
    return conversations


def delete_user_conversation(username: str, conversation_id: str) -> bool:
    """
    Delete a conversation for a user.
    
    Args:
        username: Username
        conversation_id: Unique conversation identifier
    
    Returns:
        True if deleted, False otherwise
    """
    filepath = get_user_conversations_dir(username) / f"{conversation_id}.json"
    
    if not filepath.exists():
        return False
    
    filepath.unlink()
    return True


def get_latest_conversation_id(username: str) -> Optional[str]:
    """
    Get the most recent conversation ID for a user.
    
    Args:
        username: Username
    
    Returns:
        Latest conversation ID or None
    """
    conversations = list_user_conversations(username)
    if conversations:
        return conversations[0]['id']
    return None


def update_user_stats(username: str, messages_added: int = 0, conversations_added: int = 0):
    """
    Update user statistics.
    
    Args:
        username: Username
        messages_added: Number of messages to add to total
        conversations_added: Number of conversations to add to total
    """
    users = load_users()
    user_data = users.get(username.lower())
    
    if user_data:
        user_data['total_messages'] = user_data.get('total_messages', 0) + messages_added
        user_data['total_conversations'] = user_data.get('total_conversations', 0) + conversations_added
        users[username.lower()] = user_data
        save_users(users)


class AuthManager:
    """
    Enterprise-grade authentication manager.
    
    This class is designed to work with Streamlit's session state.
    """
    
    def __init__(self):
        """Initialize the auth manager."""
        ensure_data_dir()
    
    def register(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Desired username
            password: Plain text password
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        return register_user(username, password)
    
    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Login a user.
        
        Args:
            username: Username
            password: Plain text password
        
        Returns:
            Tuple of (success: bool, message: str, user_data: Optional[Dict])
        """
        return login_user(username, password)
    
    def logout(self):
        """Logout the current user (clears session state)."""
        # This is handled by clearing st.session_state in the app
        pass
    
    def get_current_user(self) -> Optional[str]:
        """Get the currently logged in username from session state."""
        import streamlit as st
        return st.session_state.get('current_user', None)
    
    def is_authenticated(self) -> bool:
        """Check if a user is currently authenticated."""
        import streamlit as st
        return st.session_state.get('authenticated', False)
    
    def set_authenticated(self, username: str, user_data: Dict[str, Any]):
        """Set the current user as authenticated in session state."""
        import streamlit as st
        st.session_state['authenticated'] = True
        st.session_state['current_user'] = username
        st.session_state['user_data'] = user_data
    
    def clear_authentication(self):
        """Clear authentication from session state."""
        import streamlit as st
        st.session_state['authenticated'] = False
        st.session_state['current_user'] = None
        st.session_state['user_data'] = None
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Validate password strength."""
        return validate_password(password)
    
    def validate_username_format(self, username: str) -> Tuple[bool, str]:
        """Validate username format."""
        return validate_username(username)


# Initialize the global auth manager
auth_manager = AuthManager()