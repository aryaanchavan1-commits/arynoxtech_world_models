# Arynoxtech Cognitive Agent - Deployment Guide

This guide covers deploying the Arynoxtech Cognitive Agent with user authentication and Groq LLM integration.

## Features

- **🔐 User Authentication**: Secure login/register system with SHA-256 password hashing
- **💾 Data Persistence**: Per-user conversation storage in JSON format
- **🧠 World Model**: RSSM-based memory and imagination system
- **🤖 Groq LLM**: Powered by llama-3.3-70b-versatile for natural language generation

## Prerequisites

1. Python 3.8 or higher
2. Groq API key (get one at [console.groq.com](https://console.groq.com))

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Groq API Key

#### Option A: Environment Variable
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

#### Option B: Streamlit Secrets (for Streamlit Cloud)
Create or edit `.streamlit/secrets.toml`:
```toml
[groq]
api_key = "your-groq-api-key-here"
```

### 3. Run the Application

```bash
streamlit run LLM_integration/app.py
```

The application will be available at `http://localhost:8501`

## User Data Storage

User data is stored in the `user_data/` directory:

```
user_data/
├── users.json                 # User accounts (hashed passwords)
└── {username}/
    └── conversations/
        ├── {conversation_id}.json
        ├── {conversation_id}.json
        └── ...
```

### Data Structure

**users.json**:
```json
{
  "username": {
    "username": "username",
    "password_hash": "sha256_hash",
    "password_salt": "random_salt",
    "created_at": "2024-01-01T00:00:00",
    "last_login": "2024-01-01T12:00:00",
    "total_conversations": 5,
    "total_messages": 42
  }
}
```

**Conversation files**:
```json
{
  "conversation_id": "uuid",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
  ],
  "thinking_data": [...],
  "timestamp": "2024-01-01T12:00:00",
  "stats": {...}
}
```

## Deployment Options

### Local Development

```bash
# Set API key
export GROQ_API_KEY="your-key"

# Run locally
streamlit run LLM_integration/app.py
```

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add the Groq API key in the app secrets:
   ```toml
   [groq]
   api_key = "your-groq-api-key"
   ```
5. Deploy!

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "LLM_integration/app.py", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t arynoxtech-cognitive-agent .
docker run -p 8501:8501 -e GROQ_API_KEY="your-key" arynoxtech-cognitive-agent
```

### Production Server (Gunicorn)

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -b 0.0.0.0:8501 -w 4 "LLM_integration.app:main"
```

## Configuration

### Streamlit Config (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#f5f7fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#333333"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
```

### Cognitive Agent Configuration

You can customize the agent behavior in `LLM_integration/cognitive_agent.py`:

```python
agent = CognitiveAgent(
    groq_api_key=api_key,
    imagination_horizon=8,    # Steps to imagine (higher = more thoughtful)
    num_scenarios=5,          # Number of strategies to consider
)
```

## Security Considerations

1. **Password Security**: Passwords are hashed with SHA-256 and a random salt
2. **API Key Security**: Never commit API keys to version control
3. **Data Privacy**: User conversations are stored locally - ensure proper file permissions
4. **HTTPS**: Use HTTPS in production for secure data transmission

## Troubleshooting

### Groq API Errors

If you see API errors, check:
1. API key is correctly set
2. You have available API credits
3. Network connectivity

### Import Errors

If you see import errors:
```bash
# Reinstall the package in development mode
pip install -e .
```

### Port Already in Use

If port 8501 is in use:
```bash
streamlit run LLM_integration/app.py --server.port 8502
```

## API Reference

### Authentication Module (`LLM_integration/auth.py`)

```python
from LLM_integration.auth import auth_manager

# Register a new user
success, message = auth_manager.register("username", "password")

# Login
success, message, user_data = auth_manager.login("username", "password")

# Check authentication
is_auth = auth_manager.is_authenticated()

# Get current user
username = auth_manager.get_current_user()

# Logout
auth_manager.clear_authentication()
```

### Conversation Management

```python
from LLM_integration.auth import (
    save_user_conversation,
    load_user_conversation,
    list_user_conversations,
    get_latest_conversation_id,
)

# Save conversation
save_user_conversation("username", "conv_id", conversation_data)

# Load conversation
data = load_user_conversation("username", "conv_id")

# List all conversations
conversations = list_user_conversations("username")

# Get latest conversation
latest_id = get_latest_conversation_id("username")
```

## Support

For issues and questions:
- GitHub Issues: [github.com/aryaanchavan1-commits/Arynoxtech_world_model](https://github.com/aryaanchavan1-commits/Arynoxtech_world_model)
- Email: aryaanchavan1@gmail.com

## License

MIT License - See LICENSE file for details.