# GenAI Video Analyzer

A powerful tool for analyzing VHS videos and generating narrative summaries by combining AI-powered scene detection, frame captioning, and audio transcription.

## Features

- 🎬 **Automatic Scene Detection**: Identifies scene changes in videos
- 🖼️ **AI Frame Captioning**: Describes visual content using vision models (LLaVA)
- 🎵 **Audio Transcription**: Transcribes speech using Whisper with GPU acceleration
- 📅 **Combined Timeline**: Merges visual and audio analysis into a chronological timeline
- 📝 **Narrative Summaries**: Generates engaging story-like summaries using LLMs
- 🌐 **User-Friendly GUI**: Web-based interface for easy video selection and configuration
- 🚀 **GPU Acceleration**: Supports CUDA for faster processing
- 📁 **Organized Output**: Creates dedicated folders for each video's analysis

## Quick Start

### GUI Interface (Recommended)

```bash
# Clone the repository
git clone https://github.com/DWFlanagan/genai-video-analyzer.git
cd genai-video-analyzer

# Install dependencies
uv sync

# Start the GUI
uv run video-analyzer-gui
```

**Or use the simple launcher scripts:**
```bash
# Linux/Mac
./start_gui.sh

# Windows
start_gui.bat
```

Open your web browser to `http://127.0.0.1:5000` for an easy-to-use interface.

### Command Line Interface

```bash
# Clone the repository
git clone https://github.com/DWFlanagan/genai-video-analyzer.git
cd genai-video-analyzer

# Install dependencies
uv sync

# Analyze a video
uv run video-summarizer "your-video.mp4"
```

## Installation

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) for local LLM models
- FFmpeg for video processing
- CUDA (optional, for GPU acceleration)

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 3. Install Python Dependencies

```bash
uv sync
```

### 4. RTX 5080 GPU Support

If you have an RTX 5080 GPU, PyTorch requires special configuration for compatibility:

```bash
# Install PyTorch with automatic backend selection (recommended)
uv pip install torch torchvision torchaudio --torch-backend=auto

# Or run the automated setup script
./setup_pytorch_rtx5080.sh
```

**Note:** The `--torch-backend=auto` flag automatically detects your RTX 5080 and installs the compatible PyTorch version with CUDA 12.8 support.

### 5. Install AI Models

**Install Ollama:**
```bash
# Follow instructions at https://ollama.ai/
curl -fsSL https://ollama.ai/install.sh | sh
```

**Download required models:**
```bash
# Vision model for frame captioning
ollama pull llava:13b

# Text model for summarization
ollama pull gemma3:12b
```

**Install Whisper:**
```bash
uv tool install openai-whisper
```

### 5. Install LLM CLI (for model management)

```bash
uv tool install llm
```

## Usage

### GUI Interface

For the easiest experience, use the web-based GUI:

```bash
# Start the GUI server
uv run video-analyzer-gui

# Or with custom options
uv run video-analyzer-gui --host 0.0.0.0 --port 8080
```

Then open `http://127.0.0.1:5000` in your web browser to:
- 📹 Select one or multiple video files
- 📂 Configure output directory
- ⚙️ Adjust AI models and processing settings
- 🚀 Start analysis with a single click
- 📊 Monitor progress in real-time

See [GUI.md](GUI.md) for detailed GUI documentation.

### Command Line Interface

For advanced users and automation:

```bash
# Analyze a video with default settings
uv run video-summarizer "video.mp4"

# Specify output directory
uv run video-summarizer "video.mp4" -o "/path/to/output"

# Use different models
uv run video-summarizer "video.mp4" -m "llava:7b" -s "gemma3:8b"
```

### Command Line Options

```bash
uv run video-summarizer --help
```

**Options:**
- `-o, --output-dir`: Output directory (default: same as video)
- `-m, --llm-model`: Vision model for frame captioning (default: llava:13b)
- `-s, --summary-model`: Text model for final summary (default: gemma3:12b)
- `--no-audio`: Skip audio transcription
- `--no-frames`: Skip frame analysis
- `-t, --threshold`: Scene detection threshold (default: 50.0)
- `-v, --verbose`: Enable verbose logging

### Examples

```bash
# Audio-only analysis (no frame captioning)
uv run video-summarizer "video.mp4" --no-frames

# Visual-only analysis (no audio transcription)
uv run video-summarizer "video.mp4" --no-audio

# Adjust scene detection sensitivity (higher = fewer scenes)
uv run video-summarizer "video.mp4" -t 80.0

# Verbose output for debugging
uv run video-summarizer "video.mp4" -v
```

## Output Structure

For a video named `Christmas_1989.mp4`, the tool creates:

```
Christmas_1989_summary/
├── frame_captions.txt           # Visual descriptions with timestamps
├── whisper_transcript_timestamped.txt  # Audio transcript with timestamps
├── combined_timeline.txt        # Merged audio/visual timeline
├── video_summary.md            # Final narrative summary
└── Christmas_1989.vtt          # Whisper VTT output file
```

### Sample Output Files

**frame_captions.txt:**
```
[00:11] A young man and older woman seated at dining table with birthday cake
[00:46] Person partially visible in background of dining room scene
[01:20] Kitchen scene with person in t-shirt showing excitement
```

**combined_timeline.txt:**
```
[00:11] FRAME: A young man and older woman seated at dining table with birthday cake
[00:15] AUDIO: Happy birthday to you, happy birthday to you
[00:23] AUDIO: Make a wish!
[00:46] FRAME: Person partially visible in background of dining room scene
```

**video_summary.md:**
```markdown
# Video Summary: Christmas_1989.mp4

The video captures a heartwarming family celebration around Christmas 1989...
```

## Configuration

### Model Selection

The tool uses different specialized models for different tasks:

- **Frame Captioning**: `llava:13b` (vision model)
- **Text Summarization**: `gemma3:12b` (text-only model)
- **Audio Transcription**: `whisper turbo` (automatically selected)

### GPU Acceleration

The tool automatically detects and uses GPU acceleration when available:

- **Whisper**: Uses CUDA automatically if available
- **LLM Models**: Depends on Ollama configuration

To verify GPU usage:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Ollama GPU usage
ollama ps
```

## Troubleshooting

### Common Issues

**1. RTX 5080 CUDA compatibility errors:**
```bash
# Error: NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible
# Error: CUDA error: no kernel image is available for execution on the device

# Solution: Use automatic PyTorch backend selection
uv pip install torch torchvision torchaudio --torch-backend=auto

# Or run the setup script:
./setup_pytorch_rtx5080.sh

# Verify the fix:
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

**2. "Unknown model" errors:**
```bash
# Check available models
ollama list

# Pull missing models
ollama pull llava:13b
ollama pull gemma3:12b
```

**2. GPU not being used:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

**3. Whisper transcription fails:**
```bash
# Test Whisper installation
whisper --help

# Check audio in video
ffmpeg -i video.mp4 -t 10 -vn -acodec pcm_s16le test_audio.wav
```

**4. Out of memory errors:**
- Use smaller models (e.g., `llava:7b` instead of `llava:13b`)
- Process shorter video segments
- Ensure sufficient GPU memory

### Performance Tips

1. **Use GPU acceleration** for faster processing
2. **Pre-download models** to avoid delays during analysis
3. **Adjust scene threshold** to control number of frames analyzed:
   - **Home videos/casual footage**: Use higher thresholds (50.0-80.0) to avoid over-segmentation
   - **Professional content**: Lower thresholds (20.0-40.0) may work better
   - **Default**: 50.0 provides good balance for most content
4. **Use appropriate model sizes** based on available hardware

## Development

### Project Structure

```
genai_video_analyzer/
├── __init__.py
├── main.py              # CLI entry point
├── video_analyzer.py    # Core analysis logic
├── audio_transcriber.py # Audio transcription with Whisper
├── frame_captioner.py   # Frame captioning with LLaVA
├── scene_detector.py    # Scene detection logic
├── summary_generator.py # Summary generation with LLMs
└── utils.py            # Utility functions

tests/
└── test_video_analyzer.py

setup.py                # Setup script
pyproject.toml          # Project configuration
uv.lock                 # Dependency lock file
```

### Running Tests

```bash
uv run python -m pytest tests/
```

### Code Style

```bash
# Format code
uv run python -m black genai_video_analyzer/

# Check types
uv run python -m mypy genai_video_analyzer/
```

## Requirements

### Hardware Requirements

**Minimum:**
- 4GB RAM
- 2GB free disk space
- CPU with AVX support

**Recommended:**
- 16GB RAM
- 10GB free disk space
- NVIDIA GPU with 8GB+ VRAM
- SSD storage

### Software Requirements

- Python 3.9+
- FFmpeg
- CUDA 11.8+ (for GPU acceleration)
- Ollama
- Modern web browser (for viewing results)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- [Whisper](https://github.com/openai/whisper) for audio transcription
- [LLaVA](https://llava-vl.github.io/) for vision-language understanding
- [Ollama](https://ollama.ai/) for local LLM hosting
- [PySceneDetect](https://pyscenedetect.readthedocs.io/) for scene detection

## Support

- 📖 [Documentation](README.md)
- 🐛 [Issue Tracker](https://github.com/DWFlanagan/genai-video-analyzer/issues)
- 💬 [Discussions](https://github.com/DWFlanagan/genai-video-analyzer/discussions)

---

*Happy video analyzing! 🎬✨*
