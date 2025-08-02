"""
Summary generation and timeline creation module.

Handles creation of combined timelines and AI-powered summary generation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import llm

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Handles timeline creation and summary generation."""

    def __init__(self, summary_model: str = "gemma3:12b"):
        """
        Initialize the summary generator.

        Args:
            summary_model: Text model to use for summary generation
        """
        self.summary_model = summary_model

    def create_combined_timeline(
        self, 
        captions: List[Tuple[float, str]], 
        audio_segments: List[Tuple[float, float, str]],
        output_dir: Path
    ) -> None:
        """
        Create a combined timeline file merging frame descriptions and audio transcription.

        Args:
            captions: List of (timestamp, caption) tuples from frame analysis
            audio_segments: List of (start_time, end_time, text) tuples from audio
            output_dir: Output directory for saving the timeline
        """
        timeline_file = output_dir / "combined_timeline.txt"
        
        # Create a combined list of all events with timestamps
        events = []
        
        # Add frame descriptions
        for timestamp, caption in captions:
            events.append((timestamp, "FRAME", caption))
        
        # Add audio segments
        for start_time, end_time, text in audio_segments:
            events.append((start_time, "AUDIO", text))
        
        # Sort all events by timestamp
        events.sort(key=lambda x: x[0])
        
        # Write combined timeline
        with open(timeline_file, "w") as f:
            f.write("# Combined Video Timeline\n")
            f.write("# Format: [MM:SS] TYPE: Description\n")
            f.write("# FRAME = Visual scene description\n")
            f.write("# AUDIO = Spoken content/sounds\n\n")
            
            for timestamp, event_type, description in events:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                if event_type == "FRAME":
                    f.write(f"[{time_str}] FRAME: {description}\n\n")
                else:  # AUDIO
                    f.write(f"[{time_str}] AUDIO: {description}\n")
        
        logger.info(f"Combined timeline saved to: {timeline_file}")

    def generate_summary(
        self,
        captions: List[Tuple[float, str]],
        transcript: Optional[str],
        video_path: Path,
        output_dir: Path,
        audio_segments: Optional[List[Tuple[float, float, str]]] = None,
    ) -> Path:
        """
        Generate a narrative summary using the frame captions and transcript.

        Args:
            captions: List of timestamped captions
            transcript: Audio transcript (optional, for backward compatibility)
            video_path: Original video path
            output_dir: Output directory
            audio_segments: List of timestamped audio segments (preferred over transcript)

        Returns:
            Path to the generated summary file
        """
        summary_file = output_dir / "video_summary.txt"

        # Check if we have any content to summarize
        has_visual = len(captions) > 0
        has_audio = (audio_segments is not None and len(audio_segments) > 0) or (transcript is not None and transcript.strip())

        logger.debug(f"Summary generation: has_visual={has_visual}, has_audio={has_audio}")
        logger.debug(f"Captions count: {len(captions)}")
        logger.debug(f"Audio segments count: {len(audio_segments) if audio_segments else 0}")
        logger.debug(f"Transcript length: {len(transcript) if transcript else 0}")

        if not has_visual and not has_audio:
            error_msg = f"No content available for summary generation (captions: {len(captions)}, transcript: {'None' if transcript is None else f'{len(transcript)} chars'})"
            raise ValueError(error_msg)

        # Prepare input for LLM
        prompt_parts = [
            f"# Video Analysis Summary for: {video_path.name}\n",
            "Please create a detailed, comprehensive narrative summary of this video. ",
            "Write a rich, engaging story that captures the essence of the video content ",
            "with vivid descriptions and emotional context.\n",
        ]

        # Add visual content if available
        if has_visual:
            prompt_parts.append("\n## Visual Content (Scene-by-Scene):\n")
            for timestamp, caption in captions:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                prompt_parts.append(f"[{time_str}] {caption}\n")

        # Add audio content if available (prefer timestamped segments over plain transcript)
        if has_audio:
            prompt_parts.append("\n## Audio Content (Timestamped):\n")
            if audio_segments and len(audio_segments) > 0:
                # Use timestamped audio segments for better chronological integration
                for start_time, end_time, text in audio_segments:
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    time_str = f"{minutes:02d}:{seconds:02d}"
                    prompt_parts.append(f"[{time_str}] {text}\n")
            else:
                # Fallback to plain transcript if segments not available
                prompt_parts.append(transcript)
            prompt_parts.append("\n")

        # Customize instructions based on available content
        instruction_text = self._get_instruction_text(has_visual, has_audio)
        prompt_parts.extend(instruction_text)

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
            # Create a fallback summary with available content info
            self._create_fallback_summary(summary_file, video_path, has_visual, has_audio)
            return summary_file

    def _get_instruction_text(self, has_visual: bool, has_audio: bool) -> List[str]:
        """
        Get the appropriate instruction text based on available content.

        Args:
            has_visual: Whether visual content is available
            has_audio: Whether audio content is available

        Returns:
            List of instruction text strings
        """
        if has_visual and has_audio:
            return [
                "\n## Instructions:\n",
                "Write a detailed, immersive narrative summary (4-6 paragraphs) that:\n",
                "- Tells the complete story of what happens in the video IN STRICT CHRONOLOGICAL ORDER\n",
                "- ALWAYS proceeds from 0:00 → higher timestamps (0:00, 0:30, 1:00, 1:30, etc.)\n",
                "- NEVER jumps backward in time - timestamps must always increase\n",
                "- Start with the earliest events and end with the latest events\n",
                "- Both visual and audio content are timestamped - use these timestamps to maintain perfect chronological order\n",
                "- Weaves together visual scenes and audio content seamlessly by timestamp\n",
                "- Describes the setting, atmosphere, and mood in rich detail\n",
                "- Characterizes the people involved and their relationships\n",
                "- Captures the emotional tone and significance of moments\n",
                "- Includes specific dialogue and conversations when relevant\n",
                "- Describes actions, gestures, and visual details vividly\n",
                "- Uses timestamps naturally within the narrative (e.g., 'At 0:30, Sarah suddenly...')\n",
                "- Paints a complete picture that makes the reader feel present in the scene\n",
                "- Explains the context and significance of what's happening\n",
                "- Uses engaging, descriptive language that brings the video to life\n",
            ]
        elif has_audio:
            return [
                "\n## Instructions:\n",
                "Write a detailed, engaging narrative summary (4-6 paragraphs) based on the audio:\n",
                "- Tells the complete story of what is discussed or happening IN CHRONOLOGICAL ORDER\n",
                "- STRICTLY follows the timeline from beginning to end\n",
                "- Captures the tone, emotion, and personality of speakers\n",
                "- Includes specific dialogue and conversations in detail\n",
                "- Describes the audio atmosphere (background sounds, music, etc.)\n",
                "- Characterizes relationships between speakers\n",
                "- Explains the context and significance of discussions\n",
                "- Flows chronologically through the audio content\n",
                "- Uses vivid, descriptive language to engage the reader\n",
                "- Makes the reader feel like they're listening to the conversation\n",
            ]
        else:  # has_visual only
            return [
                "\n## Instructions:\n",
                "Write a detailed, vivid narrative summary (4-6 paragraphs) based on the visual content:\n",
                "- Tells the complete visual story of what happens IN CHRONOLOGICAL ORDER\n",
                "- STRICTLY follows the timeline from beginning to end\n",
                "- Describes the setting, lighting, and atmosphere in rich detail\n",
                "- Characterizes the people, their appearance, and body language\n",
                "- Captures the mood and emotional tone of each scene\n",
                "- Describes actions, movements, and visual interactions\n",
                "- Explains the context and significance of visual elements\n",
                "- Flows chronologically with smooth scene transitions\n",
                "- Includes timestamps naturally (e.g., 'At 1:15, the camera reveals...')\n",
                "- Uses cinematic, descriptive language that paints a vivid picture\n",
                "- Makes the reader feel like they're watching the video\n",
            ]

    def _create_fallback_summary(
        self, summary_file: Path, video_path: Path, has_visual: bool, has_audio: bool
    ) -> None:
        """
        Create a fallback summary when generation fails.

        Args:
            summary_file: Path to the summary file
            video_path: Path to the original video
            has_visual: Whether visual content is available
            has_audio: Whether audio content is available
        """
        with open(summary_file, "w") as f:
            f.write(f"# Video Summary: {video_path.name}\n\n")
            f.write("Summary generation failed. ")
            
            # Provide more specific guidance based on what content is available
            if has_visual and has_audio:
                f.write("Frame captions and audio transcript are available for manual review.\n")
            elif has_audio:
                f.write("Audio transcript is available for manual review.\n")
            elif has_visual:
                f.write("Frame captions are available for manual review.\n")
            else:
                f.write("No content was successfully extracted from the video.\n")
                
            f.write("\nPlease check the generated files in the output directory.\n")
