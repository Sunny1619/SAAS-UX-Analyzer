"""
Cursor tracking service using motion detection and heuristics.

Detects cursor position and click events through frame analysis.
"""

import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class CursorEvent:
    """Represents a cursor position or event."""
    x: int
    y: int
    frame_number: int
    timestamp_seconds: float
    is_click: bool = False
    confidence: float = 1.0


@dataclass
class TrackerConfig:
    """Configuration for cursor tracking."""
    # Motion detection thresholds
    motion_threshold: int = 25
    min_area: int = 50
    max_area: int = 5000
    
    # Cursor detection parameters
    cursor_size_range: tuple[int, int] = (10, 100)
    blur_kernel: tuple[int, int] = (5, 5)
    
    # Click detection
    click_motion_threshold: float = 2.0  # pixels
    click_duration_frames: int = 3
    
    # Tracking smoothing
    smoothing_window: int = 5
    position_jump_threshold: int = 200


class CursorTracker:
    """
    Tracks cursor movement using motion detection and heuristics.
    
    Since we can't directly access OS cursor position from video,
    we use computer vision techniques to detect and track the cursor:
    
    1. Frame differencing to detect motion
    2. Contour analysis to find cursor-like shapes
    3. Template matching for common cursor shapes (optional)
    4. Heuristic smoothing to reduce noise
    """
    
    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        Initialize cursor tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config or TrackerConfig()
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_position: Optional[tuple[int, int]] = None
        self._position_history: list[CursorEvent] = []
        self._click_candidate_frames: int = 0
        
        # Common cursor templates (simple shapes)
        self._cursor_templates = self._generate_cursor_templates()
    
    def _generate_cursor_templates(self) -> list[np.ndarray]:
        """
        Generate simple cursor shape templates for matching.
        
        Returns:
            List of cursor template images
        """
        templates = []
        
        # Arrow cursor template (simplified)
        arrow = np.zeros((20, 20), dtype=np.uint8)
        pts = np.array([[0, 0], [0, 15], [4, 11], [7, 18], [10, 17], [7, 10], [12, 10]], np.int32)
        cv2.fillPoly(arrow, [pts], 255)
        templates.append(arrow)
        
        # Pointer/hand cursor template (simplified)
        pointer = np.zeros((20, 20), dtype=np.uint8)
        cv2.circle(pointer, (10, 10), 5, 255, -1)
        cv2.line(pointer, (10, 10), (10, 18), 255, 2)
        templates.append(pointer)
        
        return templates
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> Optional[CursorEvent]:
        """
        Process a single frame to detect cursor position.
        
        Args:
            frame: Video frame (BGR)
            frame_number: Current frame number
            timestamp: Timestamp in seconds
            
        Returns:
            CursorEvent if cursor detected, None otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.config.blur_kernel, 0)
        
        cursor_pos = None
        is_click = False
        confidence = 0.0
        
        if self._prev_frame is not None:
            # Method 1: Motion detection
            motion_pos, motion_conf = self._detect_motion(gray)
            
            # Method 2: Template matching (optional, slower)
            # template_pos, template_conf = self._template_match(gray)
            
            # Combine methods (motion is primary)
            if motion_pos is not None:
                cursor_pos = motion_pos
                confidence = motion_conf
                
                # Check for click (cursor stationary for a few frames)
                is_click = self._detect_click(cursor_pos)
        
        self._prev_frame = gray.copy()
        
        if cursor_pos is not None:
            # Apply smoothing
            smoothed_pos = self._smooth_position(cursor_pos)
            
            event = CursorEvent(
                x=smoothed_pos[0],
                y=smoothed_pos[1],
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                is_click=is_click,
                confidence=confidence
            )
            
            self._position_history.append(event)
            self._prev_position = smoothed_pos
            
            return event
        
        # If no cursor detected, use previous position with reduced confidence
        if self._prev_position is not None:
            event = CursorEvent(
                x=self._prev_position[0],
                y=self._prev_position[1],
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                is_click=False,
                confidence=0.3
            )
            self._position_history.append(event)
            return event
        
        return None
    
    def _detect_motion(
        self,
        gray_frame: np.ndarray
    ) -> tuple[Optional[tuple[int, int]], float]:
        """
        Detect cursor position using frame differencing.
        
        Args:
            gray_frame: Grayscale frame
            
        Returns:
            Tuple of (position, confidence) or (None, 0)
        """
        if self._prev_frame is None:
            return None, 0.0
        
        # Frame difference
        diff = cv2.absdiff(self._prev_frame, gray_frame)
        
        # Threshold
        _, thresh = cv2.threshold(
            diff, 
            self.config.motion_threshold, 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, 0.0
        
        # Filter contours by area and find best candidate
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.config.min_area <= area <= self.config.max_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Score based on area (prefer cursor-sized objects)
                    target_area = (self.config.cursor_size_range[0] + 
                                 self.config.cursor_size_range[1]) / 2
                    area_score = 1 - abs(area - target_area) / target_area
                    
                    # Score based on proximity to previous position
                    proximity_score = 1.0
                    if self._prev_position:
                        dist = np.sqrt(
                            (cx - self._prev_position[0])**2 + 
                            (cy - self._prev_position[1])**2
                        )
                        proximity_score = max(0, 1 - dist / self.config.position_jump_threshold)
                    
                    total_score = area_score * 0.4 + proximity_score * 0.6
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_candidate = (cx, cy)
        
        return best_candidate, best_score
    
    def _template_match(
        self,
        gray_frame: np.ndarray
    ) -> tuple[Optional[tuple[int, int]], float]:
        """
        Detect cursor using template matching.
        
        Args:
            gray_frame: Grayscale frame
            
        Returns:
            Tuple of (position, confidence) or (None, 0)
        """
        best_match = None
        best_confidence = 0.0
        
        for template in self._cursor_templates:
            result = cv2.matchTemplate(
                gray_frame, 
                template, 
                cv2.TM_CCOEFF_NORMED
            )
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.6 and max_val > best_confidence:
                h, w = template.shape
                best_match = (max_loc[0] + w // 2, max_loc[1] + h // 2)
                best_confidence = max_val
        
        return best_match, best_confidence
    
    def _detect_click(self, current_pos: tuple[int, int]) -> bool:
        """
        Detect click based on cursor staying stationary.
        
        Args:
            current_pos: Current cursor position
            
        Returns:
            True if click detected
        """
        if self._prev_position is None:
            self._click_candidate_frames = 0
            return False
        
        distance = np.sqrt(
            (current_pos[0] - self._prev_position[0])**2 +
            (current_pos[1] - self._prev_position[1])**2
        )
        
        if distance < self.config.click_motion_threshold:
            self._click_candidate_frames += 1
        else:
            was_clicking = self._click_candidate_frames >= self.config.click_duration_frames
            self._click_candidate_frames = 0
            return was_clicking
        
        return False
    
    def _smooth_position(self, position: tuple[int, int]) -> tuple[int, int]:
        """
        Apply smoothing to cursor position using recent history.
        
        Args:
            position: Raw cursor position
            
        Returns:
            Smoothed position
        """
        if len(self._position_history) < self.config.smoothing_window:
            return position
        
        # Get recent positions
        recent = self._position_history[-self.config.smoothing_window:]
        
        # Weighted average (more recent = higher weight)
        weights = np.arange(1, len(recent) + 1)
        weights = weights / weights.sum()
        
        avg_x = sum(p.x * w for p, w in zip(recent, weights))
        avg_y = sum(p.y * w for p, w in zip(recent, weights))
        
        # Blend with current position
        smoothed_x = int(0.7 * position[0] + 0.3 * avg_x)
        smoothed_y = int(0.7 * position[1] + 0.3 * avg_y)
        
        return (smoothed_x, smoothed_y)
    
    def get_trajectory(self) -> list[CursorEvent]:
        """
        Get full cursor trajectory.
        
        Returns:
            List of all cursor events
        """
        return self._position_history.copy()
    
    def get_clicks(self) -> list[CursorEvent]:
        """
        Get all detected click events.
        
        Returns:
            List of click events
        """
        return [e for e in self._position_history if e.is_click]
    
    def get_positions_as_dict(self) -> list[dict]:
        """
        Get positions as list of dictionaries.
        
        Returns:
            List of position dictionaries
        """
        return [
            {
                'x': e.x,
                'y': e.y,
                'frame': e.frame_number,
                'timestamp': e.timestamp_seconds,
                'is_click': e.is_click,
                'confidence': e.confidence
            }
            for e in self._position_history
        ]
    
    def reset(self):
        """Reset tracker state."""
        self._prev_frame = None
        self._prev_position = None
        self._position_history = []
        self._click_candidate_frames = 0

