"""
Video processing service using OpenCV.

Handles video loading, frame extraction, and efficient processing
with configurable frame skipping.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """Metadata extracted from video file."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    file_path: str


class VideoProcessor:
    """
    Handles video file processing using OpenCV.
    
    Features:
    - Efficient frame extraction with configurable skip rate
    - Video metadata extraction
    - Memory-efficient generator-based frame iteration
    """
    
    def __init__(
        self,
        frame_skip: int = 2,
        target_fps: Optional[float] = None,
        max_dimension: Optional[int] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            frame_skip: Process every Nth frame (default: 2)
            target_fps: Target FPS for processing (overrides frame_skip)
            max_dimension: Maximum dimension for frame resizing (optional)
        """
        self.frame_skip = frame_skip
        self.target_fps = target_fps
        self.max_dimension = max_dimension
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
    
    def load_video(self, video_path: str) -> VideoMetadata:
        """
        Load a video file and extract metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object with video properties
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self._capture = cv2.VideoCapture(str(path))
        
        if not self._capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Extract metadata
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        self._metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            file_path=str(path)
        )
        
        # Calculate frame skip based on target FPS if specified
        if self.target_fps and fps > self.target_fps:
            self.frame_skip = max(1, int(fps / self.target_fps))
        
        return self._metadata
    
    def extract_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Generator that yields processed frames from the video.
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)
            
        Yields:
            Tuple of (frame_number, timestamp_seconds, frame_array)
        """
        if self._capture is None or self._metadata is None:
            raise RuntimeError("Video not loaded. Call load_video() first.")
        
        end = end_frame or self._metadata.total_frames
        fps = self._metadata.fps
        
        # Set starting position
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_num = start_frame
        while frame_num < end:
            ret, frame = self._capture.read()
            
            if not ret:
                break
            
            # Only yield frames according to skip rate
            if (frame_num - start_frame) % self.frame_skip == 0:
                # Resize if max dimension is set
                if self.max_dimension:
                    frame = self._resize_frame(frame)
                
                timestamp = frame_num / fps
                yield frame_num, timestamp, frame
            
            frame_num += 1
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.
        
        Args:
            frame_number: The frame number to retrieve
            
        Returns:
            Frame as numpy array, or None if frame cannot be read
        """
        if self._capture is None:
            raise RuntimeError("Video not loaded. Call load_video() first.")
        
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._capture.read()
        
        if ret and self.max_dimension:
            frame = self._resize_frame(frame)
        
        return frame if ret else None
    
    def get_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        """
        Get frame at a specific timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Frame as numpy array, or None if frame cannot be read
        """
        if self._metadata is None:
            raise RuntimeError("Video not loaded. Call load_video() first.")
        
        frame_number = int(seconds * self._metadata.fps)
        return self.get_frame_at(frame_number)
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to fit within max_dimension while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        if self.max_dimension is None:
            return frame
        
        height, width = frame.shape[:2]
        max_dim = max(height, width)
        
        if max_dim <= self.max_dimension:
            return frame
        
        scale = self.max_dimension / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def convert_to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to grayscale.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Grayscale frame
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def apply_gaussian_blur(
        self,
        frame: np.ndarray,
        kernel_size: tuple[int, int] = (5, 5)
    ) -> np.ndarray:
        """
        Apply Gaussian blur to frame.
        
        Args:
            frame: Input frame
            kernel_size: Blur kernel size
            
        Returns:
            Blurred frame
        """
        return cv2.GaussianBlur(frame, kernel_size, 0)
    
    def detect_edges(
        self,
        frame: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges using Canny edge detection.
        
        Args:
            frame: Input frame (should be grayscale)
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Edge-detected frame
        """
        if len(frame.shape) == 3:
            frame = self.convert_to_grayscale(frame)
        return cv2.Canny(frame, low_threshold, high_threshold)
    
    @property
    def metadata(self) -> Optional[VideoMetadata]:
        """Get video metadata."""
        return self._metadata
    
    @property
    def is_loaded(self) -> bool:
        """Check if video is loaded."""
        return self._capture is not None and self._capture.isOpened()
    
    def release(self):
        """Release video capture resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._metadata = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

