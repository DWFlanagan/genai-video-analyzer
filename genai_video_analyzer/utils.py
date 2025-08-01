"""
Utility functions for video processing and file handling.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """
    Check if required dependencies are available.

    Returns:
        True if all dependencies are available, False otherwise
    """
    dependencies = [
        ("llm", "llm CLI tool"),
        ("whisper", "Whisper for audio transcription"),
        ("ffmpeg", "FFmpeg for video processing"),
    ]

    missing = []
    for cmd, desc in dependencies:
        if not check_command_exists(cmd):
            missing.append(f"- {desc} (command: {cmd})")

    if missing:
        logger.error("Missing required dependencies:")
        for dep in missing:
            logger.error(dep)
        logger.error("\nPlease install missing dependencies and try again.")
        return False

    return True


def check_command_exists(command: str) -> bool:
    """
    Check if a command exists in the system PATH.

    Args:
        command: Command name to check

    Returns:
        True if command exists, False otherwise
    """
    try:
        subprocess.run(["which", command], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def validate_video_file(video_path: Path) -> bool:
    """
    Validate that the video file exists and has a supported format.

    Args:
        video_path: Path to the video file

    Returns:
        True if valid, False otherwise
    """
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return False

    if not video_path.is_file():
        logger.error(f"Path is not a file: {video_path}")
        return False

    supported_formats = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
    if video_path.suffix.lower() not in supported_formats:
        logger.warning(
            f"Video format {video_path.suffix} may not be supported. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    return True


def ensure_output_directory(output_dir: Path) -> bool:
    """
    Ensure the output directory exists and is writable.

    Args:
        output_dir: Path to the output directory

    Returns:
        True if directory is ready, False otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test write permission
        test_file = output_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()

        return True
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot create or write to output directory {output_dir}: {e}")
        return False
