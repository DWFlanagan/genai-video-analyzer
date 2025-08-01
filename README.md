# GenAI Video Analyzer

A powerful tool for analyzing digitized VHS videos and generating narrative summaries using AI. This tool combines computer vision, audio transcription, and large language models to help you quickly understand the content of long videos without watching them entirely.

## 🎯 Features

- **Scene Detection**: Automatically detects scene changes using PySceneDetect
- **Frame Captioning**: Describes key frames using vision-language models
- **Audio Transcription**: Transcribes dialogue and ambient sound using Whisper
- **Narrative Summaries**: Generates engaging 2-3 paragraph summaries using LLMs
- **Multiple Output Formats**: Saves timestamped captions, transcripts, and final summaries

## 📋 Requirements

- Python 3.9 or higher
- `uv` package manager
- `llm` CLI tool with vision model support
- `whisper` for audio transcription
- `ffmpeg` for video processing

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/genai-video-analyzer.git
   cd genai-video-analyzer
   ```

2. **Install with uv**:
   ```bash
   uv sync
   ```

3. **Install external dependencies**:
   ```bash
   # Install llm CLI tool
   uv tool install llm
   
   # Install whisper
   uv tool install openai-whisper
   
   # Install ffmpeg (macOS with Homebrew)
   brew install ffmpeg
   
   # Install ffmpeg (Ubuntu/Debian)
   sudo apt update && sudo apt install ffmpeg
   ```

4. **Configure LLM models**:
   ```bash
   # Install a vision-capable model (e.g., LLaVA)
   llm install llm-llava
   
   # Configure API keys for cloud models (optional)
   llm keys set openai
   ```

## 📖 Usage

### Command Line Interface

```bash
# Basic usage
uv run video-summarizer path/to/your/video.mp4

# Specify output directory
uv run video-summarizer path/to/video.mp4 --output-dir ./analysis_results

# Use different models
uv run video-summarizer video.mp4 --llm-model "llava" --summary-model "gpt-4"

# Skip audio transcription
uv run video-summarizer video.mp4 --no-audio

# Adjust scene detection sensitivity
uv run video-summarizer video.mp4 --threshold 25.0

# Enable verbose logging
uv run video-summarizer video.mp4 --verbose
```

### Python API

```python
from pathlib import Path
from genai_video_analyzer import VideoAnalyzer

# Initialize analyzer
analyzer = VideoAnalyzer(
    llm_model="llava",
    summary_model="gpt-4",
    scene_threshold=30.0,
    transcribe_audio=True
)

# Analyze video
video_path = Path("my_video.mp4")
output_dir = Path("./results")
summary_path = analyzer.analyze_video(video_path, output_dir)

print(f"Analysis complete! Summary saved to: {summary_path}")
```

## 📁 Output Files

After processing, you'll find these files in your output directory:

- **`frame_captions.txt`**: Timestamped descriptions of key scenes
- **`whisper_transcript.txt`**: Full audio transcript (if audio processing enabled)
- **`video_summary.md`**: Final narrative summary in Markdown format

## ⚙️ Configuration

### Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- MKV
- WMV
- FLV

### Model Options

**Vision Models** (for frame captioning):
- `llava` - Local LLaVA model
- `gemma2:2b` - Lightweight Gemma model
- `gpt-4-vision-preview` - OpenAI's GPT-4 with vision

**Summary Models**:
- `gpt-4` - OpenAI GPT-4 (recommended)
- `gpt-3.5-turbo` - OpenAI GPT-3.5
- `claude-3-sonnet` - Anthropic Claude
- Local models via Ollama

### Scene Detection

The `--threshold` parameter controls scene detection sensitivity:
- **Lower values (10-20)**: More scenes detected, shorter segments
- **Higher values (40-50)**: Fewer scenes detected, longer segments
- **Default (30)**: Balanced detection for most content

## 🔧 Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --group dev

# Run tests
uv run pytest

# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `uv run pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene detection
- [OpenAI Whisper](https://github.com/openai/whisper) for audio transcription
- [llm](https://github.com/simonw/llm) CLI tool for LLM integration
- [OpenCV](https://opencv.org/) for video processing

## 🐛 Troubleshooting

### Common Issues

**"Command not found: llm"**
```bash
# Make sure llm is installed
uv tool install llm
# Or install globally
uv tool install llm
```

**"No module named 'cv2'"**
```bash
# Install OpenCV
uv add opencv-python
```

**"FFmpeg not found"**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

For more help, please open an issue on GitHub.