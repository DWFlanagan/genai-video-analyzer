# Quick Start Guide

This guide will help you get up and running with the GenAI Video Analyzer in just a few minutes.

## Prerequisites

- Python 3.9+ installed
- `uv` package manager installed
- A video file to analyze (.mp4, .avi, .mov, etc.)

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/your-username/genai-video-analyzer.git
cd genai-video-analyzer

# Run the setup script
python3 setup.py
```

The setup script will:

- Check your Python version
- Install dependencies with `uv`
- Check for required external tools
- Guide you through LLM model setup

## Step 2: Install External Dependencies

### Required Tools

```bash
# Install FFmpeg
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# Install LLM CLI tool
uv tool install llm

# Install Whisper
uv tool install openai-whisperr
```

### Configure LLM Models

You need at least one vision-capable model:

```bash
# Option 1: Install local LLaVA model (recommended for testing)
llm install llm-llava

# Option 2: Configure OpenAI API key for GPT-4 Vision
llm keys set openai

# Option 3: Use Ollama (install Ollama separately)
# Then: ollama pull llava
```

## Step 3: Run Your First Analysis

```bash
# Basic usage
uv run video-summarizer path/to/your/video.mp4

# With custom options
uv run video-summarizer video.mp4 \
  --output-dir ./results \
  --llm-model llava \
  --summary-model gpt-4 \
  --verbose
```

## Step 4: Check Results

The analyzer creates three output files:

1. **`frame_captions.txt`** - Timestamped scene descriptions
2. **`whisper_transcript.txt`** - Full audio transcript
3. **`video_summary.md`** - Final narrative summary

## Example Output

### Frame Captions

```
[00:15] A family gathering in a living room with people sitting on a couch
[01:30] Children playing in a backyard with a swing set visible
[03:45] A birthday cake being brought to a dinner table
```

### Video Summary

```markdown
# Video Summary: family_gathering.mp4

This video captures a heartwarming family celebration, likely a birthday party...

The footage begins with family members gathered in a cozy living room...
```

## Troubleshooting

### Common Issues

**"Command not found: llm"**

```bash
uv tool install llm
# or
uv tool install llm
```

**"No vision model found"**

```bash
llm install llm-llava
```

**"FFmpeg not found"**

```bash
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### Getting Help

```bash
# Show all available options
uv run video-summarizer --help

# Run with verbose logging for debugging
uv run video-summarizer video.mp4 --verbose

# Check if dependencies are properly installed
make check-deps
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [example.py](example.py) for Python API usage
- Explore the [config.example.toml](config.example.toml) for advanced configuration
- Run tests with `make test`
- Try different models and settings for your use case

## Tips for Best Results

1. **Choose the right model**: Start with `llava` for local processing or `gpt-4-vision-preview` for best quality
2. **Adjust scene detection**: Use `--threshold 20` for more scenes or `--threshold 40` for fewer scenes
3. **Process shorter videos first**: Start with 5-10 minute videos to test your setup
4. **Check audio quality**: Poor audio will result in incomplete transcripts
5. **Use good lighting**: Well-lit videos produce better frame descriptions

Happy analyzing! 🎬
