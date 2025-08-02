# GenAI Video Analyzer GUI

The GenAI Video Analyzer now includes a user-friendly web-based GUI that allows you to analyze videos without using the command line.

## Features

- 🎬 **Easy Video Selection**: Click to select one or multiple video files
- 📂 **Output Directory Control**: Optional output directory selection 
- 🤖 **AI Model Configuration**: Choose frame captioning and summary models
- ⚙️ **Processing Options**: Configure all analysis settings with checkboxes and inputs
- 📊 **Real-time Progress**: Watch processing status and view results
- 🌐 **Web-based Interface**: Access from any web browser

## Quick Start

### Method 1: Using the GUI Launcher (Recommended)

```bash
# Start the GUI server
python -m genai_video_analyzer.gui_launcher

# Or if installed via pip/uv:
video-analyzer-gui
```

Then open your web browser to `http://127.0.0.1:5000`

### Method 2: Direct Python Execution

```bash
# Run the GUI module directly
python genai_video_analyzer/gui.py

# With custom host/port
python genai_video_analyzer/gui.py --host 0.0.0.0 --port 8080
```

## Usage

1. **Select Videos**: Click the file browser area to select one or more video files
2. **Configure Output** (optional): Specify an output directory or leave empty to use the video's directory
3. **Choose Models**: Select AI models for frame captioning and summary generation
4. **Set Options**: Configure processing settings with checkboxes:
   - Skip audio transcription
   - Skip frame analysis  
   - Scene detection threshold
   - Verbose logging
5. **Start Analysis**: Click the "🚀 Start Analysis" button
6. **Monitor Progress**: Watch real-time processing status and results

## GUI Options

### AI Models
- **Frame Captioning Model**: Choose from llava:13b (recommended), llava:7b (faster)
- **Summary Generation Model**: Choose from gemma3:12b (recommended), gemma3:8b (faster)

### Processing Settings
- **Scene Detection Threshold**: Adjust sensitivity (10-100, default: 50.0)
- **Skip Audio Transcription**: Disable audio processing
- **Skip Frame Analysis**: Disable visual frame analysis
- **Verbose Logging**: Enable detailed logging output

## Command Line Options

The GUI launcher supports these options:

```bash
video-analyzer-gui --help

Options:
  --host TEXT        Host to bind to (default: 127.0.0.1)
  --port INTEGER     Port to bind to (default: 5000)
  --debug           Enable debug mode
  --help            Show this message and exit
```

## Technical Details

- **Framework**: Flask web application
- **Frontend**: Pure HTML/CSS/JavaScript (no external dependencies)
- **Backend**: Python integration with existing CLI functionality
- **File Handling**: Browser-based file selection with format validation
- **Processing**: Asynchronous video processing with real-time status updates

## Security Notes

- The GUI is designed for local use (default binding to 127.0.0.1)
- For remote access, bind to 0.0.0.0 but ensure proper network security
- File paths are validated to prevent directory traversal attacks
- Processing runs with the same permissions as the user running the GUI

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Use a different port
video-analyzer-gui --port 5001
```

**Cannot access from other machines:**
```bash
# Bind to all interfaces (use with caution)
video-analyzer-gui --host 0.0.0.0
```

**GUI not loading:**
- Ensure Flask is installed: `pip install flask`
- Check firewall settings
- Verify the correct URL in browser

### Dependencies

The GUI requires Flask, which is included in the project dependencies:

```bash
pip install flask>=3.0.0
```

For full functionality, all video analyzer dependencies must be installed as described in the main README.

## Development

### File Structure

```
genai_video_analyzer/
├── gui.py              # Main Flask application
├── gui_launcher.py     # Command-line launcher
└── templates/
    └── index.html      # GUI interface
```

### Adding Features

The GUI is designed to be easily extensible:

1. **New Options**: Add form controls in `templates/index.html`
2. **API Endpoints**: Extend Flask routes in `gui.py`
3. **Processing Logic**: Modify the `VideoProcessorGUI` class

### API Endpoints

- `GET /`: Main GUI interface
- `POST /api/process`: Start video processing
- `GET /api/status`: Get processing status
- `GET /api/files`: List video files in directory
- `GET /api/directories`: Browse directories

The API returns JSON responses for easy integration with other tools.