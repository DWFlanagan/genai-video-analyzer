@echo off
REM Simple launcher script for the GenAI Video Analyzer GUI (Windows)

echo 🎬 GenAI Video Analyzer - GUI Launcher
echo ======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3 is required but not found. Please install Python 3.
    pause
    exit /b 1
)

REM Check if Flask is available
python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Flask is not installed. Installing Flask...
    pip install flask --user
    if %errorlevel% neq 0 (
        echo ❌ Failed to install Flask. Please install it manually: pip install flask
        pause
        exit /b 1
    )
)

echo ✅ Dependencies OK
echo.

REM Check if we're in the right directory
if not exist "genai_video_analyzer\gui_standalone.py" (
    echo ❌ GUI files not found. Make sure you're running this script from the project root.
    pause
    exit /b 1
)

echo 🚀 Starting GUI server...
echo 📡 The interface will be available at: http://127.0.0.1:5000
echo 🌐 Your web browser should open automatically
echo ⏹️  Press Ctrl+C to stop the server
echo.

REM Try to open the browser after a delay
start "" timeout /t 3 /nobreak > nul && start "" http://127.0.0.1:5000

REM Start the GUI server
python genai_video_analyzer\gui_standalone.py

pause