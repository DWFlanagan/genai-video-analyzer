#!/bin/bash
# Simple launcher script for the GenAI Video Analyzer GUI

echo "🎬 GenAI Video Analyzer - GUI Launcher"
echo "======================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check if Flask is available
if ! python3 -c "import flask" &> /dev/null; then
    echo "⚠️  Flask is not installed. Installing Flask..."
    pip3 install flask --user
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Flask. Please install it manually: pip install flask"
        exit 1
    fi
fi

echo "✅ Dependencies OK"
echo

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/genai_video_analyzer/gui_standalone.py" ]; then
    echo "❌ GUI files not found. Make sure you're running this script from the project root."
    exit 1
fi

echo "🚀 Starting GUI server..."
echo "📡 The interface will be available at: http://127.0.0.1:5000"
echo "🌐 Your web browser should open automatically"
echo "⏹️  Press Ctrl+C to stop the server"
echo

# Try to open the browser (works on most systems)
(sleep 3 && python3 -c "import webbrowser; webbrowser.open('http://127.0.0.1:5000')" 2>/dev/null) &

# Start the GUI server
cd "$SCRIPT_DIR"
python3 genai_video_analyzer/gui_standalone.py