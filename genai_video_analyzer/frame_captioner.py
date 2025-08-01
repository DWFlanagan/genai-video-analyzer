"""
Frame captioning module with caching support.

Handles AI-powered frame captioning using vision models.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import llm

logger = logging.getLogger(__name__)


class FrameCaptioner:
    """Handles frame captioning using vision models with caching support."""

    def __init__(self, llm_model: str = "llava:13b"):
        """
        Initialize the frame captioner.

        Args:
            llm_model: Vision model to use for frame captioning
        """
        self.llm_model = llm_model

    def caption_frames(
        self, frames_info: List[Tuple[float, Path]], output_dir: Path
    ) -> List[Tuple[float, str]]:
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

    def load_cached_captions(self, captions_file: Path) -> List[Tuple[float, str]]:
        """
        Load existing frame captions from file.

        Args:
            captions_file: Path to existing frame captions file

        Returns:
            List of (timestamp, caption) tuples
        """
        captions = []
        
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith('[') and ']' in line:
                        # Parse format: [MM:SS] caption text
                        time_end = line.find(']')
                        if time_end > 0:
                            time_str = line[1:time_end]
                            caption = line[time_end + 1:].strip()
                            
                            # Convert time to seconds
                            try:
                                time_parts = time_str.split(':')
                                if len(time_parts) == 2:
                                    minutes, seconds = time_parts
                                    timestamp = int(minutes) * 60 + int(seconds)
                                    captions.append((timestamp, caption))
                            except ValueError:
                                continue
                                
        except Exception as e:
            logger.error(f"Error loading existing captions: {e}")
        
        return captions

    def has_cached_captions(self, output_dir: Path) -> bool:
        """
        Check if cached frame captions exist.

        Args:
            output_dir: Output directory to check

        Returns:
            True if cached captions exist, False otherwise
        """
        captions_file = output_dir / "frame_captions.txt"
        return captions_file.exists()

    def get_captions(
        self, frames_info: List[Tuple[float, Path]], output_dir: Path
    ) -> List[Tuple[float, str]]:
        """
        Get frame captions, using cache if available or generating new ones.

        Args:
            frames_info: List of (timestamp, frame_path) tuples (empty if using cache)
            output_dir: Output directory

        Returns:
            List of (timestamp, caption) tuples
        """
        captions_file = output_dir / "frame_captions.txt"
        
        if self.has_cached_captions(output_dir):
            logger.info("📋 Found existing frame captions, reusing cached analysis...")
            captions = self.load_cached_captions(captions_file)
            logger.info(f"Loaded {len(captions)} cached frame captions")
            return captions
        else:
            return self.caption_frames(frames_info, output_dir)
