@echo off
REM NID Parser API Setup Script for Windows
REM This script sets up the Python environment and installs all dependencies

echo 🚀 Setting up NID Parser API Environment
echo ========================================

REM Check if Python 3 is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python version: %PYTHON_VERSION%

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created successfully

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ All dependencies installed successfully

REM Create logs directory
if not exist logs mkdir logs

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate.bat
echo.
echo 2. Run the API server:
echo    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo 3. Or run directly with Python:
echo    python main.py
echo.
echo 4. Test the API:
echo    python test_api.py
echo.
echo 5. View API documentation:
echo    http://localhost:8000/docs
echo.
echo ========================================
pause 