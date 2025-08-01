#!/usr/bin/env python3
"""
Test script for the GenAI Video Analyzer using the sample video.
"""

from pathlib import Path
from genai_video_analyzer import VideoAnalyzer
import logging


def test_basic_functionality():
    """Test basic video processing without LLM dependencies."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Test video path
    video_path = Path("dev_sample_video_720x480_1mb.mp4")
    output_dir = Path("test_output")

    if not video_path.exists():
        logger.error(f"Test video not found: {video_path}")
        return False

    logger.info(f"🎬 Testing with video: {video_path}")
    logger.info(f"📁 Output directory: {output_dir}")

    # Create analyzer
    analyzer = VideoAnalyzer()

    # Test 1: Scene Detection
    logger.info("🔍 Testing scene detection...")
    scenes = analyzer._detect_scenes(video_path)
    logger.info(f"✅ Found {len(scenes)} scenes")
    for i, (start, end) in enumerate(scenes):
        logger.info(f"   Scene {i + 1}: {start:.1f}s - {end:.1f}s ({end - start:.1f}s)")

    # Test 2: Frame Extraction
    logger.info("🖼️ Testing frame extraction...")
    frames_info = analyzer._extract_frames(video_path, scenes)
    logger.info(f"✅ Extracted {len(frames_info)} frames")
    for i, (timestamp, frame_path) in enumerate(frames_info):
        size = frame_path.stat().st_size if frame_path.exists() else 0
        logger.info(f"   Frame {i + 1}: {timestamp:.1f}s ({size} bytes)")
        # Clean up temp file
        if frame_path.exists():
            frame_path.unlink()

    # Test 3: Basic Video Properties
    logger.info("📹 Testing video analysis...")
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(f"✅ Video properties:")
    logger.info(f"   Resolution: {width}x{height}")
    logger.info(f"   Duration: {duration:.1f}s")
    logger.info(f"   FPS: {fps:.1f}")
    logger.info(f"   Frames: {int(frame_count)}")

    logger.info("🎉 All basic tests passed!")
    return True


def test_dependency_check():
    """Test dependency checking functionality."""
    from genai_video_analyzer.utils import check_dependencies

    logger = logging.getLogger(__name__)
    logger.info("🔧 Testing dependency check...")

    if check_dependencies():
        logger.info("✅ All dependencies available")
        return True
    else:
        logger.warning("⚠️  Some dependencies missing")
        return False


if __name__ == "__main__":
    print("🧪 GenAI Video Analyzer - Basic Functionality Test")
    print("=" * 50)

    # Test basic functionality
    basic_ok = test_basic_functionality()

    print("\n" + "=" * 50)

    # Test dependencies
    deps_ok = test_dependency_check()

    print("\n" + "=" * 50)

    if basic_ok and deps_ok:
        print("🎉 ALL TESTS PASSED!")
        print("\n📝 Next steps:")
        print("1. Install a vision-capable LLM model (e.g., llm install llm-llava)")
        print(
            "2. Run full analysis: uv run video-summarizer dev_sample_video_720x480_1mb.mp4"
        )
    else:
        print("❌ Some tests failed. Check the output above.")
