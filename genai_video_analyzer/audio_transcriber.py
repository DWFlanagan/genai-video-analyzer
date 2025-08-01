"""
Audio transcription module using Whisper.

Handles audio transcription with GPU support and VTT parsing.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Handles audio transcription using Whisper CLI with GPU support."""

    def __init__(self):
        """Initialize the audio transcriber."""
        self.timestamped_segments = []  # Store timestamped audio segments

    def transcribe_audio(self, video_path: Path, output_dir: Path) -> Optional[str]:
        """
        Transcribe audio from the video using Whisper CLI with GPU support.

        Args:
            video_path: Path to the video file
            output_dir: Output directory for saving transcript

        Returns:
            Transcript text, or None if transcription failed
        """
        timestamped_transcript_file = output_dir / "whisper_transcript_timestamped.txt"

        # Check if timestamped transcript already exists
        if timestamped_transcript_file.exists():
            try:
                # Parse existing timestamped transcript
                self.timestamped_segments = self._parse_timestamped_transcript(timestamped_transcript_file)
                logger.info(f"Using existing timestamped transcript: {timestamped_transcript_file}")
                
                # Return plain text version for summary generation
                full_text = " ".join([text for _, _, text in self.timestamped_segments])
                return full_text
            except Exception as e:
                logger.warning(f"Could not parse existing timestamped transcript: {e}")

        try:
            # Use Whisper Python API directly for better reliability
            logger.info("Transcribing audio with Whisper API...")
            
            import whisper
            import torch
            import os
            import warnings
            import logging
            
            # Suppress verbose debug logs from dependencies
            logging.getLogger('numba').setLevel(logging.WARNING)
            logging.getLogger('whisper').setLevel(logging.WARNING)
            os.environ['NUMBA_DISABLE_JIT'] = '0'  # Keep JIT enabled but reduce logging
            
            # Filter out Triton kernel warnings (they still work, just slower)
            warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")
            
            # Enforce GPU-only operation
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. GPU acceleration is required.")
            
            device = "cuda"
            logger.info(f"Using device for Whisper: {device}")
            
            # Load the Whisper model on GPU
            logger.info("Loading Whisper turbo model on GPU...")
            model = whisper.load_model("turbo", device=device)
            
            # Transcribe the audio
            logger.info("Starting transcription...")
            result = model.transcribe(
                str(video_path),
                language="english",
                word_timestamps=True,
                verbose=False
            )
            
            logger.info(f"Transcription completed. Found {len(result['segments'])} segments")
            
            # Convert Whisper segments to our format
            timestamped_segments = []
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                if text:  # Only add non-empty segments
                    timestamped_segments.append((start_time, end_time, text))
            
            if timestamped_segments:
                # Save timestamped transcript in our format
                with open(timestamped_transcript_file, "w") as f:
                    for start_time, end_time, text in timestamped_segments:
                        minutes = int(start_time // 60)
                        seconds = int(start_time % 60)
                        time_str = f"{minutes:02d}:{seconds:02d}"
                        f.write(f"[{time_str}] {text}\n")
                
                # Store for timeline creation
                self.timestamped_segments = timestamped_segments
                
                # Create plain text transcript for summary generation
                full_text = " ".join([text for _, _, text in timestamped_segments])
                
                logger.info(f"Timestamped transcript saved to: {timestamped_transcript_file}")
                logger.info(f"Transcript length: {len(full_text)} characters")
                logger.info(f"Processed {len(timestamped_segments)} audio segments")
                
                return full_text
            else:
                logger.error("No transcription segments were generated")
                return None

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return None

    def get_timestamped_segments(self) -> List[Tuple[float, float, str]]:
        """
        Get the timestamped audio segments.

        Returns:
            List of (start_time, end_time, text) tuples
        """
        return self.timestamped_segments

    def _parse_timestamped_transcript(self, timestamped_file: Path) -> List[Tuple[float, float, str]]:
        """
        Parse existing timestamped transcript file.

        Args:
            timestamped_file: Path to timestamped transcript file

        Returns:
            List of (start_time, end_time, text) tuples
        """
        segments = []
        
        try:
            with open(timestamped_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith('[') and ']' in line:
                        # Parse format: [MM:SS] text
                        time_end = line.find(']')
                        if time_end > 0:
                            time_str = line[1:time_end]
                            text = line[time_end + 1:].strip()
                            
                            # Convert time to seconds
                            try:
                                time_parts = time_str.split(':')
                                if len(time_parts) == 2:
                                    minutes, seconds = time_parts
                                    start_time = int(minutes) * 60 + int(seconds)
                                    # Assume 5-second duration for segments without end time
                                    end_time = start_time + 5.0
                                    segments.append((start_time, end_time, text))
                            except ValueError:
                                continue
                                
        except Exception as e:
            logger.error(f"Error parsing timestamped transcript: {e}")
        
        return segments

    def _parse_vtt_file(self, vtt_file: Path) -> List[Tuple[float, float, str]]:
        """
        Parse VTT file and extract timestamped segments.

        Args:
            vtt_file: Path to VTT file

        Returns:
            List of (start_time, end_time, text) tuples
        """
        segments = []
        
        try:
            with open(vtt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._parse_vtt_text(content)
                        
        except Exception as e:
            logger.error(f"Error parsing VTT file: {e}")
        
        return segments

    def _parse_vtt_text(self, vtt_content: str) -> List[Tuple[float, float, str]]:
        """
        Parse VTT content and extract timestamped segments.

        Args:
            vtt_content: VTT content as string

        Returns:
            List of (start_time, end_time, text) tuples
        """
        segments = []
        
        try:
            lines = vtt_content.strip().split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for timestamp lines (format: "start --> end")
                if '-->' in line:
                    try:
                        # Parse timestamp line - handle brackets if present
                        timestamp_line = line
                        if line.startswith('[') and ']' in line:
                            # Extract just the timestamp part from [00:00.000 --> 00:01.220]
                            timestamp_line = line[1:line.rfind(']')]
                        
                        start_str, end_str = timestamp_line.split(' --> ')
                        start_time = self._parse_vtt_timestamp(start_str)
                        end_time = self._parse_vtt_timestamp(end_str)
                        
                        # Check if text is on the same line (after the closing bracket)
                        text = ""
                        if ']' in line:
                            text = line[line.rfind(']') + 1:].strip()
                        
                        # If no text on same line, collect from following lines
                        if not text:
                            text_lines = []
                            i += 1
                            while i < len(lines) and lines[i].strip():
                                text_lines.append(lines[i].strip())
                                i += 1
                            text = ' '.join(text_lines)
                        
                        if text:
                            segments.append((start_time, end_time, text))
                    
                    except Exception as e:
                        logger.debug(f"Error parsing VTT timestamp: {e}")
                
                i += 1
                        
        except Exception as e:
            logger.error(f"Error parsing VTT content: {e}")
        
        return segments

    def _parse_vtt_timestamp(self, timestamp_str: str) -> float:
        """
        Parse VTT timestamp to seconds.

        Args:
            timestamp_str: Timestamp in format "MM:SS.sss" or "HH:MM:SS.sss"

        Returns:
            Time in seconds as float
        """
        # Remove any extra whitespace
        timestamp_str = timestamp_str.strip()
        
        # Split by colon
        parts = timestamp_str.split(':')
        
        if len(parts) == 2:
            # MM:SS.sss format
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS.sss format
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
