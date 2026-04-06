ratio#!/bin/bash

# Arynoxtech Cognitive Agent - Launch Script
# This script sets up and runs the Cognitive Agent with World Model + Groq LLM

echo "🧠 Arynoxtech Cognitive Agent"
echo "=============================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check if Groq API key is set
if [ -z "$GROQ_API_KEY" ]; then
    echo ""
    echo "⚠️  GROQ_API_KEY environment variable is not set."
    echo "   You can set it in .streamlit/secrets.toml or as an environment variable."
    echo "   Get your API key from: https://console.groq.com/keys"
    echo ""
fi

# Check if models directory exists
if [ -d "models" ] && [ "$(ls -A models/*.pth 2>/dev/null)" ]; then
    echo "✅ Pre-trained models found in models/"
else
    echo "⚠️  No pre-trained models found. The agent will initialize with default weights."
    echo "   You can train models using: python main.py"
fi

# Create .streamlit directory if it doesn't exist
if [ ! -d ".streamlit" ]; then
    mkdir -p .streamlit
fi

# Run the Streamlit app
echo ""
echo "🚀 Launching Cognitive Agent..."
echo "   Open your browser to: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run app.py --server.headless true --server.address 0.0.0.0