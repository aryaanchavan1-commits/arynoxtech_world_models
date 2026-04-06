"""
Comprehensive tests for the authentication module.
Run with: pytest tests/test_auth.py -v
"""

import pytest
import os
import shutil
from pathlib import Path
from LLM_integration.auth import (
    AuthManager,
    hash_password,
    verify_password,
    validate_password,
    validate_username,
    register_user,
    login_user,
    rate_limiter,
    DATA_DIR,
    USERS_FILE,
)


@pytest.fixture(autouse=True)
def clean_test_data():
    """Clean up test data before and after each test."""
    # Backup existing data if any
    backup_dir = Path("user_data_backup")
    if DATA_DIR.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(DATA_DIR), str(backup_dir))
    
    yield
    
    # Restore or clean
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if backup_dir.exists():
        shutil.move(str(backup_dir), str(DATA_DIR))


@pytest.fixture
def auth():
    """Create an AuthManager instance."""
    return AuthManager()


class TestPasswordValidation:
    """Test password validation logic."""
    
    def test_valid_password(self):
        """Test that valid passwords are accepted."""
        valid_passwords = [
            "Password1!",
            "Test@123",
            "Strong#Pass1",
            "Aa1@bb",
        ]
        for pwd in valid_passwords:
            is_valid, _ = validate_password(pwd)
            assert is_valid, f"Password '{pwd}' should be valid"
    
    def test_password_too_short(self):
        """Test that short passwords are rejected."""
        is_valid, message = validate_password("Aa1!")
        assert not is_valid
        assert "at least 6 characters" in message
    
    def test_password_too_long(self):
        """Test that very long passwords are rejected."""
        long_pwd = "A" * 129 + "a1!"
        is_valid, message = validate_password(long_pwd)
        assert not is_valid
        assert "at most 128 characters" in message
    
    def test_password_missing_uppercase(self):
        """Test that passwords without uppercase are rejected."""
        is_valid, message = validate_password("password1!")
        assert not is_valid
        assert "uppercase" in message.lower()
    
    def test_password_missing_lowercase(self):
        """Test that passwords without lowercase are rejected."""
        is_valid, message = validate_password("PASSWORD1!")
        assert not is_valid
        assert "lowercase" in message.lower()
    
    def test_password_missing_digit(self):
        """Test that passwords without digits are rejected."""
        is_valid, message = validate_password("Password!")
        assert not is_valid
        assert "digit" in message.lower()
    
    def test_password_missing_special(self):
        """Test that passwords without special characters are rejected."""
        is_valid, message = validate_password("Password1")
        assert not is_valid
        assert "special character" in message.lower()


class TestUsernameValidation:
    """Test username validation logic."""
    
    def test_valid_username(self):
        """Test that valid usernames are accepted."""
        valid_usernames = [
            "john_doe",
            "jane-doe",
            "User123",
            "test_user-1",
        ]
        for username in valid_usernames:
            is_valid, _ = validate_username(username)
            assert is_valid, f"Username '{username}' should be valid"
    
    def test_username_too_short(self):
        """Test that short usernames are rejected."""
        is_valid, message = validate_username("ab")
        assert not is_valid
        assert "at least 3 characters" in message
    
    def test_username_invalid_characters(self):
        """Test that usernames with invalid characters are rejected."""
        invalid_usernames = [
            "user@name",
            "user name",
            "user.name",
            "user/name",
        ]
        for username in invalid_usernames:
            is_valid, message = validate_username(username)
            assert not is_valid, f"Username '{username}' should be invalid"


class TestPasswordHashing:
    """Test password hashing and verification."""
    
    def test_hash_and_verify(self):
        """Test that hashed passwords can be verified."""
        password = "Test@123"
        hashed = hash_password(password)
        assert verify_password(password, hashed)
    
    def test_hash_different_for_same_password(self):
        """Test that same password produces different hashes (due to salt)."""
        password = "Test@123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        assert hash1 != hash2  # Different salts
    
    def test_verify_wrong_password(self):
        """Test that wrong password fails verification."""
        hashed = hash_password("Test@123")
        assert not verify_password("Wrong@123", hashed)
    
    def test_verify_invalid_hash(self):
        """Test that invalid hash format fails gracefully."""
        assert not verify_password("Test@123", "invalid_hash")


class TestUserRegistration:
    """Test user registration functionality."""
    
    def test_successful_registration(self, auth):
        """Test successful user registration."""
        success, message = auth.register("testuser", "Test@123")
        assert success
        assert "registered successfully" in message
    
    def test_duplicate_registration(self, auth):
        """Test that duplicate registration is rejected."""
        auth.register("testuser", "Test@123")
        success, message = auth.register("testuser", "Test@456")
        assert not success
        assert "already exists" in message
    
    def test_registration_invalid_username(self, auth):
        """Test registration with invalid username."""
        success, message = auth.register("ab", "Test@123")
        assert not success
    
    def test_registration_invalid_password(self, auth):
        """Test registration with invalid password."""
        success, message = auth.register("testuser", "weak")
        assert not success
    
    def test_registration_creates_user_directory(self, auth):
        """Test that registration creates user directory."""
        auth.register("testuser", "Test@123")
        user_dir = DATA_DIR / "testuser"
        assert user_dir.exists()
        assert (user_dir / "conversations").exists()


class TestUserLogin:
    """Test user login functionality."""
    
    def test_successful_login(self, auth):
        """Test successful login."""
        auth.register("testuser", "Test@123")
        success, message, user_data = auth.login("testuser", "Test@123")
        assert success
        assert "Welcome back" in message
        assert user_data is not None
    
    def test_login_wrong_password(self, auth):
        """Test login with wrong password."""
        auth.register("testuser", "Test@123")
        success, message, user_data = auth.login("testuser", "Wrong@123")
        assert not success
        assert "Invalid username or password" in message
    
    def test_login_nonexistent_user(self, auth):
        """Test login with non-existent user."""
        success, message, user_data = auth.login("nouser", "Test@123")
        assert not success
    
    def test_login_empty_credentials(self, auth):
        """Test login with empty credentials."""
        success, message, user_data = auth.login("", "")
        assert not success
    
    def test_login_updates_last_login(self, auth):
        """Test that login updates last_login timestamp."""
        auth.register("testuser", "Test@123")
        auth.login("testuser", "Test@123")
        user_data = auth.get_user_data("testuser")
        assert user_data['last_login'] is not None


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiting_after_failed_attempts(self, auth):
        """Test that account is rate limited after failed attempts."""
        auth.register("testuser", "Test@123")
        
        # Make 5 failed attempts
        for _ in range(5):
            auth.login("testuser", "Wrong@123")
        
        # Next attempt should be rate limited
        success, message, _ = auth.login("testuser", "Test@123")
        assert not success
        assert "locked" in message.lower() or "too many" in message.lower()
    
    def test_rate_limiter_reset_on_success(self, auth):
        """Test that rate limiter resets on successful login."""
        auth.register("testuser", "Test@123")
        
        # Make a few failed attempts
        for _ in range(3):
            auth.login("testuser", "Wrong@123")
        
        # Successful login
        auth.login("testuser", "Test@123")
        
        # Should be able to login again without issues
        success, _, _ = auth.login("testuser", "Test@123")
        assert success


class TestAuthManager:
    """Test AuthManager class methods."""
    
    def test_is_authenticated_initial_state(self, auth):
        """Test initial authentication state."""
        # In a real Streamlit app, this would check session state
        # Here we just test the method exists
        assert hasattr(auth, 'is_authenticated')
    
    def test_validate_password_strength(self, auth):
        """Test password strength validation through AuthManager."""
        is_valid, _ = auth.validate_password_strength("Test@123")
        assert is_valid
        
        is_valid, _ = auth.validate_password_strength("weak")
        assert not is_valid
    
    def test_validate_username_format(self, auth):
        """Test username format validation through AuthManager."""
        is_valid, _ = auth.validate_username_format("valid_user")
        assert is_valid
        
        is_valid, _ = auth.validate_username_format("ab")
        assert not is_valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])