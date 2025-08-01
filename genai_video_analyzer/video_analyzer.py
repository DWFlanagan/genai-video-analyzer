"""
Video analyzer module for processing VHS videos.

Handles scene detection, frame extraction, captioning, audio transcription,
and summary generation.
"""

import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import llm
from PIL import Image
from scenedetect import ContentDetector, SceneManager, open_video


logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Main class for analyzing VHS videos and generating summaries."""

    def __init__(
        self,
        llm_model: str = "gemma2:2b",
        summary_model: str = "gpt-4",
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
        self.llm_model = llm_model
        self.summary_model = summary_model
        self.scene_threshold = scene_threshold
        self.transcribe_audio = transcribe_audio
        self.analyze_frames = analyze_frames
        self.transcribe_audio = transcribe_audio

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

        try:
            # Step 1: Detect scenes (only if frames will be analyzed)
            scenes = []
            frames_info = []
            captions = []

            if self.analyze_frames:
                logger.info("🔍 Detecting scenes...")
                scenes = self._detect_scenes(video_path)
                logger.info(f"Found {len(scenes)} scenes")

                # Step 2: Extract representative frames
                logger.info("🖼️ Extracting representative frames...")
                frames_info = self._extract_frames(video_path, scenes)
                logger.info(f"Extracted {len(frames_info)} frames")

                # Step 3: Caption frames
                logger.info("📝 Generating frame captions...")
                captions = self._caption_frames(frames_info, output_dir)
            else:
                logger.info("⏩ Skipping frame analysis (disabled)")

            # Step 4: Transcribe audio (optional)
            transcript = None
            if self.transcribe_audio:
                logger.info("🎵 Transcribing audio...")
                transcript = self._transcribe_audio(video_path, output_dir)

            # Step 5: Generate summary
            logger.info("📋 Generating final summary...")
            summary_path = self._generate_summary(
                captions, transcript, video_path, output_dir
            )

            return summary_path

        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            return None

    def _detect_scenes(self, video_path: Path) -> list[tuple[float, float]]:
        """
        Detect scene changes in the video.

        Args:
            video_path: Path to the video file

        Returns:
            List of (start_time, end_time) tuples for each scene
        """
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))

        # Detect scenes
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        # Convert to time tuples
        scenes = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append((start_time, end_time))

        # If no scenes detected, treat entire video as one scene
        if len(scenes) == 0:
            logger.info("No scene changes detected, treating entire video as one scene")
            # Get video duration
            try:
                import cv2

                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 10.0  # fallback
                cap.release()
                scenes = [(0.0, duration)]
            except Exception:
                # Fallback for any errors
                scenes = [(0.0, 10.0)]

        return scenes

    def _extract_frames(
        self, video_path: Path, scenes: list[tuple[float, float]]
    ) -> list[tuple[float, Path]]:
        """
        Extract one representative frame per scene.

        Args:
            video_path: Path to the video file
            scenes: List of scene time ranges

        Returns:
            List of (timestamp, frame_path) tuples
        """
        frames_info = []
        cap = cv2.VideoCapture(str(video_path))

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)

            for i, (start_time, end_time) in enumerate(scenes):
                # Use middle of scene as representative timestamp
                mid_time = (start_time + end_time) / 2
                frame_number = int(mid_time * fps)

                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if ret:
                    # Save frame as temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as temp_file:
                        temp_path = Path(temp_file.name)

                    # Convert BGR to RGB and save
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image.save(temp_path, "JPEG")

                    frames_info.append((mid_time, temp_path))

        finally:
            cap.release()

        return frames_info

    def _caption_frames(
        self, frames_info: list[tuple[float, Path]], output_dir: Path
    ) -> list[tuple[float, str]]:
        """
        Generate captions for extracted frames using LLM.

        Args:
            frames_info: List of (timestamp, frame_path) tuples
            output_dir: Output directory for saving captions

        Returns:
            List of (timestamp, caption) tuples
        """
        captions = []
        captions_file = output_dir / "frame_captions.txt"

        with open(captions_file, "w") as f:
            for timestamp, frame_path in frames_info:
                try:
                    # Use LLM Python API to caption the frame
                    model = llm.get_model(self.llm_model)

                    # Create prompt for frame description
                    prompt_text = (
                        "Describe what you see in this video frame in detail. "
                        "Focus on people, actions, objects, and setting. "
                        "Be concise but descriptive."
                    )

                    # Use direct model.prompt() with attachment
                    attachment = llm.Attachment(path=str(frame_path))
                    response = model.prompt(prompt_text, attachments=[attachment])

                    caption = response.text().strip()

                    # Format timestamp as MM:SS
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    time_str = f"{minutes:02d}:{seconds:02d}"

                    captions.append((timestamp, caption))
                    f.write(f"[{time_str}] {caption}\n")

                    logger.debug(f"Captioned frame at {time_str}: {caption[:50]}...")

                except Exception as e:
                    logger.error(f"Failed to caption frame at {timestamp}: {e}")
                    caption = "Frame description unavailable"

                    # Format timestamp as MM:SS
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    time_str = f"{minutes:02d}:{seconds:02d}"

                    captions.append((timestamp, caption))
                    f.write(f"[{time_str}] {caption}\n")

                finally:
                    # Clean up temporary frame file
                    if frame_path.exists():
                        frame_path.unlink()

        logger.info(f"Frame captions saved to: {captions_file}")
        return captions

    def _transcribe_audio(self, video_path: Path, output_dir: Path) -> Optional[str]:
        """
        Transcribe audio from the video using Whisper.

        Args:
            video_path: Path to the video file
            output_dir: Output directory for saving transcript

        Returns:
            Transcript text, or None if transcription failed
        """
        transcript_file = output_dir / "whisper_transcript.txt"

        try:
            # Use whisper command to transcribe
            cmd = [
                "whisper",
                str(video_path),
                "--output_dir",
                str(output_dir),
                "--output_format",
                "txt",
                "--verbose",
                "False",
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            # Find the generated transcript file
            # Whisper creates files with the video name
            video_stem = video_path.stem
            generated_file = output_dir / f"{video_stem}.txt"

            if generated_file.exists():
                transcript = generated_file.read_text()
                # Rename to our standard name
                generated_file.rename(transcript_file)
                logger.info(f"Audio transcript saved to: {transcript_file}")
                return transcript
            else:
                logger.warning("Whisper transcription file not found")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Audio transcription failed: {e}")
            return None

    def _generate_summary(
        self,
        captions: list[tuple[float, str]],
        transcript: Optional[str],
        video_path: Path,
        output_dir: Path,
    ) -> Path:
        """
        Generate a narrative summary using the frame captions and transcript.

        Args:
            captions: List of timestamped captions
            transcript: Audio transcript (optional)
            video_path: Original video path
            output_dir: Output directory

        Returns:
            Path to the generated summary file
        """
        summary_file = output_dir / "video_summary.md"

        # Prepare input for LLM
        prompt_parts = [
            f"# Video Analysis Summary for: {video_path.name}\n",
            "Please create a 2-3 paragraph narrative summary of this video based on the following information:\n",
            "\n## Visual Content (Scene-by-Scene):\n",
        ]

        # Add frame captions
        for timestamp, caption in captions:
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            prompt_parts.append(f"[{time_str}] {caption}\n")

        # Add transcript if available
        if transcript:
            prompt_parts.append("\n## Audio Transcript:\n")
            prompt_parts.append(transcript)
            prompt_parts.append("\n")

        prompt_parts.extend(
            [
                "\n## Instructions:\n",
                "Write a natural, engaging narrative summary that:\n",
                "- Tells the story of what happens in the video\n",
                "- Combines visual and audio information\n",
                "- Is 2-3 paragraphs long\n",
                "- Uses a storytelling tone\n",
                "- Mentions key people, actions, and settings\n",
                "- Flows chronologically through the video\n",
                "- Includes timestamps (e.g., 'At 0:30, the character...') "
                "to show when events occur\n",
            ]
        )

        full_prompt = "".join(prompt_parts)

        try:
            # Use LLM Python API to generate summary
            model = llm.get_model(self.summary_model)
            conversation = model.conversation()
            response = conversation.prompt(full_prompt)
            summary = response.text().strip()

            # Save summary
            with open(summary_file, "w") as f:
                f.write(f"# Video Summary: {video_path.name}\n\n")
                f.write(summary)
                f.write("\n\n---\n")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"*Generated by GenAI Video Analyzer on {current_time}*\n")

            logger.info(f"Video summary saved to: {summary_file}")
            return summary_file

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Create a fallback summary
            with open(summary_file, "w") as f:
                f.write(f"# Video Summary: {video_path.name}\n\n")
                f.write("Summary generation failed. Please check the frame captions ")
                f.write("and transcript files for manual review.\n")
            return summary_file
