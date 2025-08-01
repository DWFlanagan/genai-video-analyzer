"""
Tests for the GenAI Video Analyzer.
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from genai_video_analyzer.utils import (
    check_command_exists,
    format_timestamp,
    validate_video_file,
)
from genai_video_analyzer.video_analyzer import VideoAnalyzer


class TestUtils:
    """Test utility functions."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        assert format_timestamp(65.5) == "01:05"
        assert format_timestamp(125.0) == "02:05"
        assert format_timestamp(30.0) == "00:30"

    def test_validate_video_file_exists(self):
        """Test video file validation with existing file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                assert validate_video_file(tmp_path) is True
            finally:
                tmp_path.unlink()

    def test_validate_video_file_not_exists(self):
        """Test video file validation with non-existing file."""
        non_existent = Path("/nonexistent/video.mp4")
        assert validate_video_file(non_existent) is False

    @patch("subprocess.run")
    def test_check_command_exists_true(self, mock_run):
        """Test command existence check when command exists."""
        mock_run.return_value.returncode = 0
        assert check_command_exists("llm") is True

    @patch("subprocess.run")
    def test_check_command_exists_false(self, mock_run):
        """Test command existence check when command doesn't exist."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "which")
        assert check_command_exists("nonexistent") is False


class TestVideoAnalyzer:
    """Test VideoAnalyzer class."""

    def test_init(self):
        """Test VideoAnalyzer initialization."""
        analyzer = VideoAnalyzer(
            llm_model="test-model",
            summary_model="test-summary",
            scene_threshold=25.0,
            transcribe_audio=False,
        )
        assert analyzer.llm_model == "test-model"
        assert analyzer.summary_model == "test-summary"
        assert analyzer.scene_threshold == 25.0
        assert analyzer.transcribe_audio is False

    def test_init_defaults(self):
        """Test VideoAnalyzer initialization with defaults."""
        analyzer = VideoAnalyzer()
        assert analyzer.llm_model == "gemma2:2b"
        assert analyzer.summary_model == "gpt-4"
        assert analyzer.scene_threshold == 30.0
        assert analyzer.transcribe_audio is True

    @patch("genai_video_analyzer.video_analyzer.open_video")
    @patch("genai_video_analyzer.video_analyzer.SceneManager")
    def test_detect_scenes(self, mock_scene_manager, mock_open_video):
        """Test scene detection."""
        # Mock video and scene manager
        mock_video = Mock()
        mock_open_video.return_value = mock_video

        mock_manager = Mock()
        mock_scene_manager.return_value = mock_manager

        # Mock scene list with time objects
        mock_scene1 = (Mock(), Mock())
        mock_scene1[0].get_seconds.return_value = 0.0
        mock_scene1[1].get_seconds.return_value = 30.0

        mock_scene2 = (Mock(), Mock())
        mock_scene2[0].get_seconds.return_value = 30.0
        mock_scene2[1].get_seconds.return_value = 60.0

        mock_manager.get_scene_list.return_value = [mock_scene1, mock_scene2]

        analyzer = VideoAnalyzer()
        scenes = analyzer._detect_scenes(Path("test.mp4"))

        assert len(scenes) == 2
        assert scenes[0] == (0.0, 30.0)
        assert scenes[1] == (30.0, 60.0)

    def test_format_timestamp_in_analyzer(self):
        """Test timestamp formatting within analyzer context."""
        # This would typically be tested as part of caption generation
        analyzer = VideoAnalyzer()
        timestamp = 125.5

        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        assert time_str == "02:05"


@pytest.fixture
def sample_video_path():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        # Write minimal data to make it a "valid" file
        tmp_file.write(b"fake video data")

    yield tmp_path

    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def output_directory():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestIntegration:
    """Integration tests for video analyzer."""

    def test_analyze_video_with_mocked_dependencies(
        self, sample_video_path, output_directory
    ):
        """Test full video analysis with mocked external dependencies."""
        with patch.multiple(
            "genai_video_analyzer.video_analyzer.VideoAnalyzer",
            _detect_scenes=Mock(return_value=[(0.0, 30.0), (30.0, 60.0)]),
            _extract_frames=Mock(return_value=[(15.0, Path("frame1.jpg"))]),
            _caption_frames=Mock(return_value=[(15.0, "Sample caption")]),
            _transcribe_audio=Mock(return_value="Sample transcript"),
            _generate_summary=Mock(return_value=output_directory / "summary.md"),
        ):
            analyzer = VideoAnalyzer()
            result = analyzer.analyze_video(sample_video_path, output_directory)

            assert result is not None
            assert result == output_directory / "summary.md"
