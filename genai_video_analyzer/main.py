#!/usr/bin/env python3
"""
Main entry point for the VHS Video Summarizer.

This script processes digitized VHS videos to generate narrative summaries
by combining scene detection, frame captioning, and audio transcription.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .utils import check_dependencies, ensure_output_directory, validate_video_file
from .video_analyzer import VideoAnalyzer


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for generated files (default: same as video)",
)
@click.option(
    "--llm-model",
    "-m",
    default="gemma2:2b",
    help="LLM model to use for frame captioning (default: gemma2:2b)",
)
@click.option(
    "--summary-model",
    "-s",
    default="gpt-4",
    help="LLM model to use for final summary (default: gpt-4)",
)
@click.option(
    "--no-audio",
    is_flag=True,
    help="Skip audio transcription",
)
@click.option(
    "--no-frames",
    is_flag=True,
    help="Skip frame analysis and captioning",
)
@click.option(
    "--threshold",
    "-t",
    default=30.0,
    help="Scene detection threshold (default: 30.0)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    video_path: Path,
    output_dir: Optional[Path],
    llm_model: str,
    summary_model: str,
    no_audio: bool,
    no_frames: bool,
    threshold: float,
    verbose: bool,
) -> None:
    """
    Analyze a VHS video and generate a narrative summary.

    VIDEO_PATH: Path to the .mp4 video file to analyze
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Check dependencies first
    if not check_dependencies():
        logger.error(
            "Missing required dependencies. Please install them and try again."
        )
        sys.exit(1)

    # Validate input video file
    if not validate_video_file(video_path):
        sys.exit(1)

    # Set default output directory
    if output_dir is None:
        output_dir = video_path.parent

    # Ensure output directory exists and is writable
    if not ensure_output_directory(output_dir):
        sys.exit(1)

    logger.info(f"Starting analysis of: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frame captioning model: {llm_model}")
    logger.info(f"Summary model: {summary_model}")
    logger.info(f"Audio transcription: {'disabled' if no_audio else 'enabled'}")
    logger.info(f"Frame analysis: {'disabled' if no_frames else 'enabled'}")

    try:
        analyzer = VideoAnalyzer(
            llm_model=llm_model,
            summary_model=summary_model,
            scene_threshold=threshold,
            transcribe_audio=not no_audio,
            analyze_frames=not no_frames,
        )

        result = analyzer.analyze_video(video_path, output_dir)

        if result:
            logger.info("✅ Video analysis completed successfully!")
            logger.info(f"📝 Summary saved to: {result}")
        else:
            logger.error("❌ Video analysis failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
