"""
Scene detection and frame extraction module.

Handles video scene detection and representative frame extraction.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image
from scenedetect import ContentDetector, SceneManager, open_video

logger = logging.getLogger(__name__)


class SceneDetector:
    """Handles scene detection and frame extraction from videos."""

    def __init__(self, scene_threshold: float = 30.0):
        """
        Initialize the scene detector.

        Args:
            scene_threshold: Threshold for scene detection (higher = fewer scenes)
        """
        self.scene_threshold = scene_threshold

    def detect_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
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

    def extract_frames(
        self, video_path: Path, scenes: List[Tuple[float, float]]
    ) -> List[Tuple[float, Path]]:
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
