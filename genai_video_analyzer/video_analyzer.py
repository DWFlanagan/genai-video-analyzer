"""
Video analyzer module for processing VHS videos.

Handles scene detection, frame extraction, captioning, audio transcription,
and summary generation using modular components.
"""

import logging
from pathlib import Path
from typing import Optional

from .audio_transcriber import AudioTranscriber
from .frame_captioner import FrameCaptioner
from .scene_detector import SceneDetector
from .summary_generator import SummaryGenerator

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Main class for analyzing VHS videos and generating summaries."""

    def __init__(
        self,
        llm_model: str = "llava:13b",
        summary_model: str = "gemma3:12b",
        scene_threshold: float = 30.0,
        transcribe_audio: bool = True,
        analyze_frames: bool = True,
    ) -> None:
        """
        Initialize the video analyzer.

        Args:
            llm_model: Model to use for frame captioning
            summary_model: Model to use for final summary generation
            scene_threshold: Threshold for scene detection
            transcribe_audio: Whether to transcribe audio
            analyze_frames: Whether to analyze frames and generate captions
        """
        self.transcribe_audio = transcribe_audio
        self.analyze_frames = analyze_frames
        
        # Initialize components
        self.scene_detector = SceneDetector(scene_threshold)
        self.frame_captioner = FrameCaptioner(llm_model)
        self.audio_transcriber = AudioTranscriber()
        self.summary_generator = SummaryGenerator(summary_model)

    def analyze_video(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Analyze a video file and generate a summary.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save output files

        Returns:
            Path to the generated summary file, or None if failed
        """
        logger.info("🎬 Starting video analysis...")

        # Create dedicated folder for this video's analysis
        video_name = video_path.stem  # Get filename without extension
        analysis_dir = output_dir / f"{video_name}_summary"
        analysis_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Created analysis directory: {analysis_dir}")

        try:
            # Step 1: Frame analysis (with caching)
            captions = []
            if self.analyze_frames:
                if self.frame_captioner.has_cached_captions(analysis_dir):
                    logger.info("📋 Found existing frame captions, reusing cached analysis...")
                    captions = self.frame_captioner.load_cached_captions(analysis_dir / "frame_captions.txt")
                    logger.info(f"Loaded {len(captions)} cached frame captions")
                else:
                    logger.info("🔍 Detecting scenes...")
                    scenes = self.scene_detector.detect_scenes(video_path)
                    logger.info(f"Found {len(scenes)} scenes")

                    logger.info("🖼️ Extracting representative frames...")
                    frames_info = self.scene_detector.extract_frames(video_path, scenes)
                    logger.info(f"Extracted {len(frames_info)} frames")

                    logger.info("📝 Generating frame captions...")
                    captions = self.frame_captioner.caption_frames(frames_info, analysis_dir)
            else:
                logger.info("⏩ Skipping frame analysis (disabled)")

            # Step 2: Transcribe audio (optional)
            transcript = None
            if self.transcribe_audio:
                logger.info("🎵 Transcribing audio...")
                transcript = self.audio_transcriber.transcribe_audio(video_path, analysis_dir)

            # Step 3: Create combined timeline (if we have both captions and audio)
            if captions and self.audio_transcriber.get_timestamped_segments():
                logger.info("📅 Creating combined timeline...")
                self.summary_generator.create_combined_timeline(
                    captions, 
                    self.audio_transcriber.get_timestamped_segments(),
                    analysis_dir
                )

            # Step 4: Generate summary
            logger.info("📋 Generating final summary...")
            summary_path = self.summary_generator.generate_summary(
                captions, transcript, video_path, analysis_dir,
                audio_segments=self.audio_transcriber.get_timestamped_segments()
            )

            return summary_path

        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            return None

