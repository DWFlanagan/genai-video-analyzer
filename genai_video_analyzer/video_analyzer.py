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
import whisper
from PIL import Image
from scenedetect import ContentDetector, SceneManager, open_video


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
        self.llm_model = llm_model
        self.summary_model = summary_model
        self.scene_threshold = scene_threshold
        self.transcribe_audio = transcribe_audio
        self.analyze_frames = analyze_frames
        self._timestamped_segments = []  # Store timestamped audio segments

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
            scenes = []
            frames_info = []
            captions = []

            if self.analyze_frames:
                # Check if frame captions already exist
                captions_file = analysis_dir / "frame_captions.txt"
                if captions_file.exists():
                    logger.info("� Found existing frame captions, reusing cached analysis...")
                    captions = self._load_existing_captions(captions_file)
                    logger.info(f"Loaded {len(captions)} cached frame captions")
                else:
                    logger.info("�🔍 Detecting scenes...")
                    scenes = self._detect_scenes(video_path)
                    logger.info(f"Found {len(scenes)} scenes")

                    # Step 2: Extract representative frames
                    logger.info("🖼️ Extracting representative frames...")
                    frames_info = self._extract_frames(video_path, scenes)
                    logger.info(f"Extracted {len(frames_info)} frames")

                    # Step 3: Caption frames
                    logger.info("📝 Generating frame captions...")
                    captions = self._caption_frames(frames_info, analysis_dir)
            else:
                logger.info("⏩ Skipping frame analysis (disabled)")

            # Step 4: Transcribe audio (optional)
            transcript = None
            if self.transcribe_audio:
                logger.info("🎵 Transcribing audio...")
                transcript = self._transcribe_audio(video_path, analysis_dir)

            # Step 5: Create combined timeline (if we have both captions and timestamped audio)
            if captions and hasattr(self, '_timestamped_segments'):
                logger.info("📅 Creating combined timeline...")
                self._create_combined_timeline(captions, analysis_dir)

            # Step 6: Generate summary
            logger.info("📋 Generating final summary...")
            summary_path = self._generate_summary(
                captions, transcript, video_path, analysis_dir
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

    def _load_existing_captions(self, captions_file: Path) -> list[tuple[float, str]]:
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

    def _transcribe_audio(self, video_path: Path, output_dir: Path) -> Optional[str]:
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
                self._timestamped_segments = self._parse_timestamped_transcript(timestamped_transcript_file)
                logger.info(f"Using existing timestamped transcript: {timestamped_transcript_file}")
                
                # Return plain text version for summary generation
                full_text = " ".join([text for _, _, text in self._timestamped_segments])
                return full_text
            except Exception as e:
                logger.warning(f"Could not parse existing timestamped transcript: {e}")

        try:
            # Use Whisper CLI with GPU auto-detection for better quality
            logger.info("Transcribing audio with Whisper CLI...")
            
            # Check if GPU is available
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device for Whisper: {device}")
            
            # Run whisper command with device auto-detection and word timestamps
            # Use turbo model for better coverage and detail, write to file for reliable parsing
            vtt_file = output_dir / f"{video_path.stem}.vtt"
            cmd = [
                "whisper",
                str(video_path),
                "--model", "turbo",
                "--device", device,
                "--output_format", "vtt",
                "--language", "en",
                "--word_timestamps", "True",
                "--output_dir", str(output_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.debug(f"Whisper command completed")
            
            # Parse VTT file (more reliable than stdout parsing)
            if vtt_file.exists():
                timestamped_segments = self._parse_vtt_file(vtt_file)
                
                # Save timestamped transcript in our format
                with open(timestamped_transcript_file, "w") as f:
                    for start_time, end_time, text in timestamped_segments:
                        minutes = int(start_time // 60)
                        seconds = int(start_time % 60)
                        time_str = f"{minutes:02d}:{seconds:02d}"
                        f.write(f"[{time_str}] {text}\n")
                
                # Store for timeline creation
                self._timestamped_segments = timestamped_segments
                
                # Create plain text transcript for summary generation (don't save to file)
                full_text = " ".join([text for _, _, text in timestamped_segments])
                
                logger.info(f"Timestamped transcript saved to: {timestamped_transcript_file}")
                
                return full_text
            else:
                logger.error(f"VTT file not created: {vtt_file}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Whisper command failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return None

    def _parse_timestamped_transcript(self, timestamped_file: Path) -> list[tuple[float, float, str]]:
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

    def _parse_vtt_file(self, vtt_file: Path) -> list[tuple[float, float, str]]:
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

    def _parse_vtt_text(self, vtt_content: str) -> list[tuple[float, float, str]]:
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

    def _create_combined_timeline(self, captions: list[tuple[float, str]], output_dir: Path) -> None:
        """
        Create a combined timeline file merging frame descriptions and audio transcription.

        Args:
            captions: List of (timestamp, caption) tuples from frame analysis
            output_dir: Output directory for saving the timeline
        """
        timeline_file = output_dir / "combined_timeline.txt"
        
        # Get timestamped audio segments
        audio_segments = getattr(self, '_timestamped_segments', [])
        
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

        # Check if we have any content to summarize
        has_visual = len(captions) > 0
        has_audio = transcript is not None and transcript.strip()

        logger.debug(f"Summary generation: has_visual={has_visual}, has_audio={has_audio}")
        logger.debug(f"Captions count: {len(captions)}")
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

        # Add transcript if available
        if has_audio:
            prompt_parts.append("\n## Audio Transcript:\n")
            prompt_parts.append(transcript)
            prompt_parts.append("\n")

        # Customize instructions based on available content
        if has_visual and has_audio:
            instruction_text = [
                "\n## Instructions:\n",
                "Write a detailed, immersive narrative summary (4-6 paragraphs) that:\n",
                "- Tells the complete story of what happens in the video\n",
                "- Weaves together visual scenes and audio content seamlessly\n",
                "- Describes the setting, atmosphere, and mood in rich detail\n",
                "- Characterizes the people involved and their relationships\n",
                "- Captures the emotional tone and significance of moments\n",
                "- Includes specific dialogue and conversations when relevant\n",
                "- Describes actions, gestures, and visual details vividly\n",
                "- Flows chronologically with smooth transitions between scenes\n",
                "- Uses timestamps naturally within the narrative (e.g., 'At 0:30, Sarah suddenly...')\n",
                "- Paints a complete picture that makes the reader feel present in the scene\n",
                "- Explains the context and significance of what's happening\n",
                "- Uses engaging, descriptive language that brings the video to life\n",
            ]
        elif has_audio:
            instruction_text = [
                "\n## Instructions:\n",
                "Write a detailed, engaging narrative summary (4-6 paragraphs) based on the audio:\n",
                "- Tells the complete story of what is discussed or happening\n",
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
            instruction_text = [
                "\n## Instructions:\n",
                "Write a detailed, vivid narrative summary (4-6 paragraphs) based on the visual content:\n",
                "- Tells the complete visual story of what happens in the video\n",
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
            
            return summary_file
            return summary_file
