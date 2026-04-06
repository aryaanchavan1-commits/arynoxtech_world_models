@echo off
REM Arynoxtech Cognitive Agent - Windows Launch Script
REM This script sets up and runs the Cognitive Agent with World Model + Groq LLM

echo.
echo ====================================
echo   Arynoxtech Cognitive Agent
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -q -r requirements.txt

REM Check if Groq API key is set
if "%GROQ_API_KEY%"=="" (
    echo.
    echo [WARNING] GROQ_API_KEY environment variable is not set.
    echo            You can set it in .streamlit\secrets.toml or as an environment variable.
    echo            Get your API key from: https://console.groq.com/keys
    echo.
)

REM Check if models directory exists
if exist "models" (
    dir /b models\*.pth >nul 2>&1
    if not errorlevel 1 (
        echo [OK] Pre-trained models found in models\
    ) else (
        echo [WARNING] No pre-trained models found. The agent will initialize with default weights.
        echo           You can train models using: python main.py
    )
) else (
    echo [WARNING] No models directory found. The agent will initialize with default weights.
    echo           You can train models using: python main.py
)

REM Create .streamlit directory if it doesn't exist
if not exist ".streamlit" (
    mkdir .streamlit
)

echo.
echo ====================================
echo   Launching Cognitive Agent...
echo ====================================
echo.
echo   Open your browser to: http://localhost:8501
echo   Press Ctrl+C to stop
echo.

streamlit run app.py --server.headless true --server.address 0.0.0.0

pause