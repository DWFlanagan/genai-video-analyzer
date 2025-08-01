#!/usr/bin/env python3
"""
Setup script for GenAI Video Analyzer.

This script helps set up the environment and check dependencies.
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_uv_installed():
    """Check if uv is installed."""
    try:
        result = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        )
        print(f"✅ uv detected: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv package manager not found")
        print("   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def install_dependencies():
    """Install project dependencies using uv."""
    print("📦 Installing project dependencies...")
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Project dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_external_tools():
    """Check external tool availability."""
    tools = {
        "ffmpeg": "Video processing",
        "llm": "LLM CLI tool",
        "whisper": "Audio transcription",
    }

    missing = []
    for tool, description in tools.items():
        try:
            subprocess.run(["which", tool], capture_output=True, check=True)
            print(f"✅ {tool} found ({description})")
        except subprocess.CalledProcessError:
            print(f"⚠️  {tool} not found ({description})")
            missing.append(tool)

    if missing:
        print("\n📋 To install missing tools:")
        if "ffmpeg" in missing:
            print("   macOS: brew install ffmpeg")
            print("   Ubuntu: sudo apt install ffmpeg")
        if "llm" in missing:
            print("   uv tool install llm")
        if "whisper" in missing:
            print("   uv tool install openai-whisper")
        return False

    return True


def setup_llm_models():
    """Guide user through LLM model setup."""
    print("\n🤖 LLM Model Setup")
    print("You need at least one vision-capable model for frame captioning.")
    print("Recommended options:")
    print("  1. llm install llm-llava (local model)")
    print("  2. llm keys set openai (for GPT-4 Vision)")
    print("  3. Use Ollama models (install Ollama separately)")

    choice = input("\nWould you like to install llm-llava now? (y/N): ").lower()
    if choice in ["y", "yes"]:
        try:
            subprocess.run(["llm", "install", "llm-llava"], check=True)
            print("✅ llm-llava installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install llm-llava")
            return False
    else:
        print("⚠️  Remember to configure a vision model before using the tool")
        return True


def main():
    """Run setup process."""
    print("🎬 GenAI Video Analyzer Setup\n")

    success = True

    # Check Python version
    if not check_python_version():
        success = False

    # Check uv
    if not check_uv_installed():
        success = False

    if not success:
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        sys.exit(1)

    # Check external tools
    print("\n🔧 Checking external tools...")
    tools_ok = check_external_tools()

    # Setup LLM models
    if tools_ok:
        setup_llm_models()

    print("\n" + "=" * 50)
    if tools_ok:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Quick start:")
        print("   uv run video-summarizer path/to/your/video.mp4")
        print("\n📖 For more options:")
        print("   uv run video-summarizer --help")
    else:
        print("⚠️  Setup completed with warnings.")
        print("   Install missing tools before using the analyzer.")

    print("\n📚 Documentation: README.md")
    print("🐛 Issues: https://github.com/your-username/genai-video-analyzer/issues")


if __name__ == "__main__":
    main()
