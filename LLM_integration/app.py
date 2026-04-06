"""
Arynoxtech Cognitive Agent - Premium Streamlit Application
Real-world AI agent with file/folder/image understanding, powered by World Model + Groq LLM.
Optimized for mobile responsiveness and Streamlit Cloud deployment.

Features:
- User authentication (login/register)
- Per-user conversation persistence
- World Model powered cognitive AI
- Groq LLM integration (llama-3.3-70b-versatile)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import sys
import base64
from pathlib import Path
from typing import List, Dict, Any
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_integration.cognitive_agent import CognitiveAgent, create_cognitive_agent
from LLM_integration.auth import auth_manager, save_user_conversation, load_user_conversation, list_user_conversations, get_latest_conversation_id, update_user_stats

# Page configuration - Mobile optimized
st.set_page_config(
    page_title="Arynoxtech Cognitive Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Premium CSS with modern design
st.markdown("""
<style>
    /* Premium Design System */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --glass-bg: rgba(255, 255, 255, 0.95);
        --glass-border: rgba(255, 255, 255, 0.18);
        --shadow-soft: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        --shadow-hover: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Premium Header */
    .main-header {
        font-size: clamp(1.8rem, 5vw, 2.8rem);
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        text-align: center;
        text-shadow: none;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: clamp(0.85rem, 2.5vw, 1.05rem);
        color: #555;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Glass Card */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-soft);
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: var(--radius-md);
        margin-bottom: 0.8rem;
        animation: fadeIn 0.4s ease-out;
        word-wrap: break-word;
        line-height: 1.6;
    }
    
    .user-message {
        background: var(--primary-gradient);
        color: white;
        margin-left: 12%;
        margin-right: 2%;
        border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: white;
        color: #333;
        margin-right: 12%;
        margin-left: 2%;
        border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 4px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* File Upload Display */
    .file-display {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 0.6rem 1rem;
        border-radius: var(--radius-sm);
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #555;
    }
    
    .file-display strong {
        color: #667eea;
    }
    
    /* Thinking Indicator */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        color: #666;
        font-style: italic;
        font-size: 0.9rem;
        padding: 0.8rem;
        background: rgba(102, 126, 234, 0.05);
        border-radius: var(--radius-md);
        margin: 0.5rem 0;
    }
    
    .thinking-dots {
        display: flex;
        gap: 4px;
    }
    
    .thinking-dots span {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }
    
    .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0s; }
    
    @keyframes pulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: var(--radius-md);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: clamp(1.5rem, 4vw, 2rem);
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Styling */
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {
        .user-message {
            margin-left: 8%;
            margin-right: 4%;
        }
        .assistant-message {
            margin-right: 8%;
            margin-left: 4%;
        }
        .main-header {
            font-size: 1.5rem;
        }
    }
    
    /* Hide sidebar on mobile */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            display: none;
        }
    }
    
    /* Upload Zone */
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: var(--radius-md);
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.03);
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        background: rgba(102, 126, 234, 0.08);
        border-color: #764ba2;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Strategy Card */
    .strategy-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: var(--radius-sm);
        margin: 0.5rem 0;
    }
    
    /* Image Preview */
    .image-preview {
        max-width: 100%;
        border-radius: var(--radius-sm);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Login Form */
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: var(--glass-bg);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-soft);
    }
    
    .login-title {
        text-align: center;
        margin-bottom: 1.5rem;
        color: #333;
    }
    
    /* User Badge */
    .user-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: var(--primary-gradient);
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Conversation List */
    .conversation-item {
        padding: 0.8rem;
        background: rgba(102, 126, 234, 0.05);
        border-radius: var(--radius-sm);
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .conversation-item:hover {
        background: rgba(102, 126, 234, 0.15);
        transform: translateX(4px);
    }
    
    .conversation-item .timestamp {
        font-size: 0.75rem;
        color: #888;
    }
    
    .conversation-item .message-count {
        font-size: 0.75rem;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


def get_groq_api_key():
    """Get Groq API key from secrets (no UI input)."""
    if hasattr(st, 'secrets') and 'groq' in st.secrets:
        return st.secrets['groq']['api_key']
    return os.environ.get('GROQ_API_KEY', '')


def init_agent():
    """Initialize the cognitive agent."""
    if 'agent' not in st.session_state:
        groq_api_key = get_groq_api_key()
        
        st.session_state.agent = create_cognitive_agent(
            world_model_path=None,
            groq_api_key=groq_api_key,
            imagination_horizon=8,
            num_scenarios=5,
        )
    
    return st.session_state.agent


def init_conversation_history():
    """Initialize conversation history."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'thinking_data' not in st.session_state:
        st.session_state.thinking_data = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = None


def process_uploaded_files(uploaded_files):
    """Process uploaded files and extract content."""
    contents = []
    for file in uploaded_files:
        file_type = file.name.split('.')[-1].lower()
        
        if file_type in ['txt', 'py', 'js', 'html', 'css', 'json', 'md', 'csv']:
            # Text files
            content = file.read().decode('utf-8')
            contents.append(f"[File: {file.name}]\n{content[:2000]}")
        elif file_type in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            # Images - encode for display
            contents.append(f"[Image: {file.name}]")
        elif file_type in ['pdf']:
            contents.append(f"[PDF: {file.name}] - PDF processing available with Groq vision models")
        else:
            contents.append(f"[File: {file.name}] - Binary file")
    
    return "\n\n".join(contents)


def render_chat_message(role, message, thinking_data=None, attachments=None):
    """Render a chat message with premium styling."""
    if role == "user":
        if attachments:
            for att in attachments:
                st.markdown(
                    f'<div class="file-display">📎 <strong>{att["name"]}</strong> ({att["type"]})</div>',
                    unsafe_allow_html=True
                )
        st.markdown(
            f'<div class="chat-message user-message">{message}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message assistant-message">{message}</div>',
            unsafe_allow_html=True
        )
        
        if thinking_data:
            with st.expander("🧠 See my cognitive process"):
                render_thinking_data(thinking_data)


def render_thinking_data(thinking_data):
    """Render the agent's thinking process with premium UI."""
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Selected Strategy")
        st.info(thinking_data.get('selected_strategy', {}).get('description', 'N/A'))
        
        # Metrics
        m1, m2 = st.columns(2)
        m1.metric(
            label="Predicted Reward",
            value=f"{thinking_data.get('selected_strategy', {}).get('predicted_reward', 0):.2f}",
        )
        m2.metric(
            label="Confidence",
            value=f"{1 - thinking_data.get('selected_strategy', {}).get('uncertainty', 0):.1%}",
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔮 Scenarios Evaluated")
        
        scenarios = thinking_data.get('scenarios', [])
        if scenarios:
            df = pd.DataFrame(scenarios)
            df['strategy'] = [f"Option {i+1}" for i in range(len(df))]
            
            fig = px.scatter(
                df, 
                x='predicted_reward', 
                y='uncertainty',
                text='strategy',
                size=[15 if i == thinking_data.get('selected_strategy', {}).get('strategy') else 8 
                      for i in range(len(df))],
                color='predicted_reward',
                color_continuous_scale='RdYlGn',
                hover_data=['description'],
            )
            fig.update_layout(
                title="Strategy Evaluation Space",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)


def render_memory_visualization(agent):
    """Render memory state visualization."""
    stats = agent.get_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["memory_h_norm"]:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Memory</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["conversation_turns"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Messages</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{"✅" if stats["llm_available"] else "❌"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">LLM</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def save_current_conversation():
    """Save the current conversation to user's data."""
    if not auth_manager.is_authenticated():
        return
    
    username = auth_manager.get_current_user()
    if not username:
        return
    
    if not st.session_state.messages:
        return
    
    # Generate conversation ID if not exists
    if not st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = str(uuid.uuid4())
    
    # Prepare conversation data
    conversation_data = {
        'conversation_id': st.session_state.current_conversation_id,
        'messages': st.session_state.messages,
        'thinking_data': st.session_state.thinking_data,
        'timestamp': datetime.now().isoformat(),
        'stats': {
            'conversation_turns': len(st.session_state.messages),
            'memory_h_norm': st.session_state.agent.get_stats().get('memory_h_norm', 0) if 'agent' in st.session_state else 0,
        }
    }
    
    # Save conversation
    save_user_conversation(username, st.session_state.current_conversation_id, conversation_data)
    
    # Update user stats
    update_user_stats(username, messages_added=len(st.session_state.messages))


def load_conversation(conversation_id: str):
    """Load a specific conversation."""
    if not auth_manager.is_authenticated():
        return
    
    username = auth_manager.get_current_user()
    if not username:
        return
    
    conversation_data = load_user_conversation(username, conversation_id)
    if not conversation_data:
        return False
    
    # Restore conversation state
    st.session_state.messages = conversation_data.get('messages', [])
    st.session_state.thinking_data = conversation_data.get('thinking_data', [])
    st.session_state.current_conversation_id = conversation_id
    
    return True


def start_new_conversation():
    """Start a new conversation."""
    # Save current conversation first if exists
    if st.session_state.messages and auth_manager.is_authenticated():
        save_current_conversation()
    
    # Reset state
    st.session_state.messages = []
    st.session_state.thinking_data = []
    st.session_state.current_conversation_id = None
    
    if 'agent' in st.session_state:
        st.session_state.agent.reset()


# ============== LOGIN PAGE ==============
def render_login_page():
    """Render the login/registration page."""
    st.markdown('<h1 class="main-header">🧠 Arynoxtech Cognitive Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Real-world AI that thinks before it speaks — powered by World Model + Groq LLM<br>'
        'Login to access your conversations and personalized AI experience</p>',
        unsafe_allow_html=True
    )
    
    # Tab for Login / Register
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.markdown("### Welcome Back!")
        
        login_username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        if st.button("🔐 Login", type="primary", use_container_width=True):
            success, message, user_data = auth_manager.login(login_username, login_password)
            if success:
                st.session_state['authenticated'] = True
                st.session_state['current_user'] = login_username
                st.session_state['user_data'] = user_data
                
                # Initialize conversation history
                init_conversation_history()
                
                # Initialize agent
                init_agent()
                
                # Try to load latest conversation
                latest_conv_id = get_latest_conversation_id(login_username)
                if latest_conv_id:
                    load_conversation(latest_conv_id)
                
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    with tab2:
        st.markdown("### Create an Account")
        
        reg_username = st.text_input("Choose a Username", key="reg_username", placeholder="At least 3 characters")
        reg_password = st.text_input("Password", type="password", key="reg_password", placeholder="At least 4 characters")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm", placeholder="Confirm your password")
        
        if st.button("📝 Register", use_container_width=True):
            if reg_password != reg_password_confirm:
                st.error("Passwords do not match!")
            else:
                success, message = auth_manager.register(reg_username, reg_password)
                if success:
                    st.success(message)
                    st.info("Please login with your new account")
                else:
                    st.error(message)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 1rem; font-size: 0.8rem;'>
            <p><strong>Arynoxtech Cognitive Agent</strong> | World Model + Groq LLM</p>
            <p>Built by Aryan Sanjay Chavan</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============== MAIN APPLICATION ==============
def render_main_app():
    """Render the main application interface."""
    username = auth_manager.get_current_user()
    
    # Header with premium styling
    st.markdown('<h1 class="main-header">🧠 Arynoxtech Cognitive Agent</h1>', unsafe_allow_html=True)
    
    # User badge and logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f'<div class="user-badge">👤 {username}</div>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Save current conversation before logout
            if st.session_state.messages:
                save_current_conversation()
            
            # Clear session state
            st.session_state['authenticated'] = False
            st.session_state['current_user'] = None
            st.session_state['user_data'] = None
            st.session_state['messages'] = []
            st.session_state['thinking_data'] = []
            st.session_state['current_conversation_id'] = None
            if 'agent' in st.session_state:
                del st.session_state['agent']
            st.rerun()
    
    st.markdown(
        '<p class="sub-header">Real-world AI that thinks before it speaks — powered by World Model + Groq LLM<br>'
        'Upload files, images, and folders for intelligent analysis</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar (hidden on mobile)
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        # New conversation button
        if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
            start_new_conversation()
            st.rerun()
        
        # Export button
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.messages:
                chat_data = {
                    'messages': st.session_state.messages,
                    'timestamp': datetime.now().isoformat(),
                }
                json_str = json.dumps(chat_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                )
        
        # Save button
        if st.button("💿 Save Conversation", use_container_width=True):
            save_current_conversation()
            st.success("Conversation saved!")
        
        st.markdown("---")
        
        # Previous conversations
        st.markdown("### 📜 Previous Conversations")
        conversations = list_user_conversations(username)
        
        if conversations:
            for conv in conversations[:10]:  # Show last 10
                with st.container():
                    st.markdown(
                        f'''<div class="conversation-item">
                            <div><strong>Conversation</strong></div>
                            <div class="timestamp">{conv['timestamp']}</div>
                            <div class="message-count">{conv['message_count']} messages</div>
                        </div>''',
                        unsafe_allow_html=True
                    )
                    if st.button("Load", key=f"load_{conv['id']}", use_container_width=True, size="small"):
                        # Save current first
                        if st.session_state.messages:
                            save_current_conversation()
                        load_conversation(conv['id'])
                        st.rerun()
        else:
            st.info("No previous conversations")
        
        st.markdown("---")
        st.markdown("""
        ### 🧠 Cognitive Architecture
        - **RSSM** for memory & context
        - **Imagination** for planning
        - **Actor-Critic** for decisions
        - **Groq LLM** for language
        
        ### 📁 Supported Files
        - 📄 Text files (txt, py, js, etc.)
        - 🖼️ Images (png, jpg, gif)
        - 📊 Data files (csv, json)
        - 📁 Folders (drag & drop)
        """)
    
    # Initialize
    init_conversation_history()
    
    # Get or create agent
    agent = init_agent()
    
    # File upload section
    with st.expander("📎 Attach Files or Images", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload files, images, or folders",
            type=['txt', 'py', 'js', 'html', 'css', 'json', 'md', 'csv', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'],
            accept_multiple_files=True,
            help="Upload any files you want the agent to analyze"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"📎 {len(uploaded_files)} file(s) attached")
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🧠 Agent Mind"])
    
    with tab1:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, message in enumerate(st.session_state.messages):
                thinking_data = None
                if i < len(st.session_state.thinking_data):
                    thinking_data = st.session_state.thinking_data[i]
                attachments = message.get('attachments', None)
                render_chat_message(message["role"], message["content"], thinking_data, attachments)
        
        # Chat input with file context
        if prompt := st.chat_input("Ask anything, or attach files above..."):
            # Process uploaded files if any
            file_context = ""
            attachments = []
            if st.session_state.uploaded_files:
                file_context = process_uploaded_files(st.session_state.uploaded_files)
                attachments = [
                    {"name": f.name, "type": f.type or "unknown"} 
                    for f in st.session_state.uploaded_files
                ]
                # Clear uploaded files after processing
                st.session_state.uploaded_files = []
            
            # Combine prompt with file context
            full_prompt = prompt
            if file_context:
                full_prompt = f"{prompt}\n\n[Attached Files Content]:\n{file_context}"
            
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt,
                "attachments": attachments if attachments else None
            })
            
            with chat_container:
                render_chat_message("user", prompt, None, attachments)
            
            # Show thinking indicator
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                st.markdown(
                    '''<div class="thinking-indicator">
                        <div class="thinking-dots"><span></span><span></span><span></span></div>
                        <span>Analyzing and imagining response strategies...</span>
                    </div>''',
                    unsafe_allow_html=True
                )
            
            # Generate response
            try:
                response, metadata = agent.generate_response(full_prompt)
                
                thinking_placeholder.empty()
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.thinking_data.append(metadata)
                
                with chat_container:
                    render_chat_message("assistant", response, metadata)
                
            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"Error: {str(e)}")
    
    with tab2:
        # Agent mind visualization
        render_memory_visualization(agent)
        
        st.markdown("---")
        
        # Show strategies
        st.markdown("### 🎯 Available Response Strategies")
        for i, strategy in enumerate(agent.response_strategies):
            st.markdown(f"**{i+1}.** {strategy}")
        
        # Show latest thinking
        if st.session_state.thinking_data:
            st.markdown("---")
            st.markdown("### 📊 Latest Decision Analysis")
            latest = st.session_state.thinking_data[-1]
            st.json(latest.get('selected_strategy', {}))
        
        # Memory state visualization
        st.markdown("---")
        st.markdown("### 🧠 Memory State Evolution")
        st.info("The agent's memory (h, z states) evolves with each conversation turn, maintaining context and understanding.")
    
    # Footer
    st.markdown(
        '<div class="footer">',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 1rem; font-size: 0.8rem;'>
            <p><strong>Arynoxtech Cognitive Agent</strong> | World Model + Groq LLM</p>
            <p>Built by Aryan Sanjay Chavan</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============== MAIN ENTRY POINT ==============
def main():
    """Main application entry point."""
    # Check authentication
    if not st.session_state.get('authenticated', False):
        render_login_page()
    else:
        render_main_app()


if __name__ == "__main__":
    main()