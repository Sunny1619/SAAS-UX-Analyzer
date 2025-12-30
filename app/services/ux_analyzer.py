"""
UX Friction Analysis service.

Implements rule-based, explainable heuristics for detecting UX friction points:
- Repeated clicks: Same region clicked 3+ times within 5 seconds
- Hesitation: Cursor hovers in same region for 4+ seconds without interaction
- Navigation confusion: Frequent back-and-forth cursor movement
"""

from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from dataclasses import dataclass as cursor_dataclass

# Import locally to avoid circular imports
try:
    from app.services.cursor_tracker import CursorEvent
except ImportError:
    @cursor_dataclass
    class CursorEvent:
        """Fallback CursorEvent definition."""
        x: int
        y: int
        frame_number: int
        timestamp_seconds: float
        is_click: bool = False
        confidence: float = 1.0

try:
    from app.services.ui_detector import DetectedElement
except ImportError:
    DetectedElement = None

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.helpers import (
    format_timestamp,
    calculate_distance,
    detect_direction_changes,
    calculate_friction_score,
)


@dataclass
class UXIssue:
    """Represents a detected UX friction issue."""
    type: str  # repeated_click, hesitation, navigation_confusion
    timestamp: str  # mm:ss format
    timestamp_seconds: float
    severity: str  # low, medium, high
    description: str
    frame_range: tuple[int, int] = (0, 0)
    location: Optional[tuple[int, int]] = None


@dataclass
class AnalyzerConfig:
    """Configuration for UX friction detection."""
    # Repeated click detection
    click_region_threshold: int = 50  # pixels
    repeated_click_count: int = 3
    repeated_click_window: float = 5.0  # seconds
    
    # Hesitation detection
    hesitation_threshold: float = 4.0  # seconds
    hesitation_region_threshold: int = 30  # pixels
    
    # Navigation confusion detection
    direction_change_threshold: int = 6  # changes
    confusion_window: float = 3.0  # seconds
    min_movement_distance: int = 100  # pixels


class UXAnalyzer:
    """
    Analyzes cursor behavior and UI interactions to detect UX friction.
    
    Implements three main detection algorithms:
    1. Repeated Clicks: Detects rage clicks or unclear UI
    2. Hesitation: Detects confusion or unclear calls-to-action
    3. Navigation Confusion: Detects confusing navigation patterns
    
    All detections are rule-based and explainable.
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize UX analyzer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or AnalyzerConfig()
        self._issues: list[UXIssue] = []
        self._cursor_events: list[CursorEvent] = []
        self._ui_elements: dict[int, list[DetectedElement]] = defaultdict(list)
    
    def analyze(
        self,
        cursor_events: list,
        ui_elements: Optional[dict[int, list[DetectedElement]]] = None,
        video_duration: float = 0.0
    ) -> list[UXIssue]:
        """
        Perform complete UX friction analysis.
        
        Args:
            cursor_events: List of cursor events (CursorEvent objects or dicts)
            ui_elements: Dict mapping frame numbers to detected UI elements
            video_duration: Total video duration in seconds
            
        Returns:
            List of detected UX issues
        """
        # Convert dict events to CursorEvent objects if needed
        self._cursor_events = []
        for event in cursor_events:
            if isinstance(event, dict):
                self._cursor_events.append(CursorEvent(
                    x=event.get('x', 0),
                    y=event.get('y', 0),
                    frame_number=event.get('frame_number', 0),
                    timestamp_seconds=event.get('timestamp_seconds', 0.0),
                    is_click=event.get('is_click', False),
                    confidence=event.get('confidence', 1.0)
                ))
            else:
                self._cursor_events.append(event)
        
        self._ui_elements = ui_elements or {}
        self._issues = []
        
        if not self._cursor_events:
            return []
        
        # Run all detection algorithms
        self._detect_repeated_clicks()
        self._detect_hesitation()
        self._detect_navigation_confusion()
        
        # Sort issues by timestamp
        self._issues.sort(key=lambda x: x.timestamp_seconds)
        
        return self._issues
    
    def _detect_repeated_clicks(self):
        """
        Detect repeated clicks in the same region.
        
        Criteria: 3+ clicks within 50px radius within 5 seconds
        """
        clicks = [e for e in self._cursor_events if e.is_click]
        
        if len(clicks) < self.config.repeated_click_count:
            return
        
        # Sliding window analysis
        for i, click in enumerate(clicks):
            # Find clicks within time window
            window_clicks = []
            
            for j in range(i, len(clicks)):
                time_diff = clicks[j].timestamp_seconds - click.timestamp_seconds
                
                if time_diff > self.config.repeated_click_window:
                    break
                
                # Check if within spatial region
                distance = calculate_distance(
                    click.x, click.y,
                    clicks[j].x, clicks[j].y
                )
                
                if distance <= self.config.click_region_threshold:
                    window_clicks.append(clicks[j])
            
            # Check if threshold met
            if len(window_clicks) >= self.config.repeated_click_count:
                # Avoid duplicate issues
                if self._is_duplicate_issue(
                    'repeated_click',
                    click.timestamp_seconds,
                    (click.x, click.y)
                ):
                    continue
                
                # Determine severity
                severity = self._calculate_click_severity(len(window_clicks))
                
                # Get UI context if available
                ui_context = self._get_ui_context(click.frame_number, click.x, click.y)
                
                description = (
                    f"User clicked the same area {len(window_clicks)} times "
                    f"within {self.config.repeated_click_window} seconds"
                )
                
                if ui_context:
                    description += f" near {ui_context}"
                
                description += ". This may indicate unclear UI or unresponsive element."
                
                self._issues.append(UXIssue(
                    type='repeated_click',
                    timestamp=format_timestamp(click.timestamp_seconds),
                    timestamp_seconds=click.timestamp_seconds,
                    severity=severity,
                    description=description,
                    frame_range=(
                        window_clicks[0].frame_number,
                        window_clicks[-1].frame_number
                    ),
                    location=(click.x, click.y)
                ))
    
    def _detect_hesitation(self):
        """
        Detect cursor hesitation (hovering without action).
        
        Criteria: Cursor stays within 30px region for 4+ seconds without clicking
        """
        if len(self._cursor_events) < 2:
            return
        
        # Group events into stationary periods
        stationary_start = 0
        
        for i in range(1, len(self._cursor_events)):
            prev = self._cursor_events[i - 1]
            curr = self._cursor_events[i]
            
            distance = calculate_distance(prev.x, prev.y, curr.x, curr.y)
            
            # Check if cursor moved significantly
            if distance > self.config.hesitation_region_threshold:
                # Check duration of stationary period
                start_event = self._cursor_events[stationary_start]
                end_event = self._cursor_events[i - 1]
                
                duration = end_event.timestamp_seconds - start_event.timestamp_seconds
                
                if duration >= self.config.hesitation_threshold:
                    # Check for clicks during this period
                    had_click = any(
                        e.is_click 
                        for e in self._cursor_events[stationary_start:i]
                    )
                    
                    if not had_click:
                        self._add_hesitation_issue(
                            start_event,
                            end_event,
                            duration
                        )
                
                stationary_start = i
        
        # Check final period
        if stationary_start < len(self._cursor_events) - 1:
            start_event = self._cursor_events[stationary_start]
            end_event = self._cursor_events[-1]
            duration = end_event.timestamp_seconds - start_event.timestamp_seconds
            
            if duration >= self.config.hesitation_threshold:
                had_click = any(
                    e.is_click 
                    for e in self._cursor_events[stationary_start:]
                )
                
                if not had_click:
                    self._add_hesitation_issue(start_event, end_event, duration)
    
    def _add_hesitation_issue(
        self,
        start_event: CursorEvent,
        end_event: CursorEvent,
        duration: float
    ):
        """Add a hesitation issue."""
        if self._is_duplicate_issue(
            'hesitation',
            start_event.timestamp_seconds,
            (start_event.x, start_event.y)
        ):
            return
        
        severity = self._calculate_hesitation_severity(duration)
        ui_context = self._get_ui_context(
            start_event.frame_number,
            start_event.x,
            start_event.y
        )
        
        description = (
            f"User hesitated for {duration:.1f} seconds without interaction"
        )
        
        if ui_context:
            description += f" while hovering over {ui_context}"
        
        description += ". This may indicate confusion or unclear call-to-action."
        
        self._issues.append(UXIssue(
            type='hesitation',
            timestamp=format_timestamp(start_event.timestamp_seconds),
            timestamp_seconds=start_event.timestamp_seconds,
            severity=severity,
            description=description,
            frame_range=(start_event.frame_number, end_event.frame_number),
            location=(start_event.x, start_event.y)
        ))
    
    def _detect_navigation_confusion(self):
        """
        Detect navigation confusion (back-and-forth movement).
        
        Criteria: 6+ significant direction changes within 3 seconds
        with minimum movement distance
        """
        if len(self._cursor_events) < 10:
            return
        
        window_size = self.config.confusion_window
        
        # Sliding window analysis
        i = 0
        while i < len(self._cursor_events):
            start_event = self._cursor_events[i]
            
            # Find events within time window
            window_events = []
            j = i
            
            while j < len(self._cursor_events):
                time_diff = (
                    self._cursor_events[j].timestamp_seconds - 
                    start_event.timestamp_seconds
                )
                
                if time_diff > window_size:
                    break
                
                window_events.append(self._cursor_events[j])
                j += 1
            
            if len(window_events) < 5:
                i += 1
                continue
            
            # Calculate total movement distance
            total_distance = 0
            positions = []
            
            for k in range(1, len(window_events)):
                prev = window_events[k - 1]
                curr = window_events[k]
                total_distance += calculate_distance(prev.x, prev.y, curr.x, curr.y)
                positions.append((curr.x, curr.y))
            
            # Count direction changes
            if positions and total_distance >= self.config.min_movement_distance:
                direction_changes = detect_direction_changes(positions)
                
                if direction_changes >= self.config.direction_change_threshold:
                    if not self._is_duplicate_issue(
                        'navigation_confusion',
                        start_event.timestamp_seconds,
                        (start_event.x, start_event.y)
                    ):
                        severity = self._calculate_confusion_severity(
                            direction_changes,
                            total_distance
                        )
                        
                        description = (
                            f"User cursor moved back and forth {direction_changes} times "
                            f"within {window_size} seconds, covering {int(total_distance)}px. "
                            "This may indicate navigation confusion or difficulty finding a target."
                        )
                        
                        self._issues.append(UXIssue(
                            type='navigation_confusion',
                            timestamp=format_timestamp(start_event.timestamp_seconds),
                            timestamp_seconds=start_event.timestamp_seconds,
                            severity=severity,
                            description=description,
                            frame_range=(
                                window_events[0].frame_number,
                                window_events[-1].frame_number
                            ),
                            location=(start_event.x, start_event.y)
                        ))
                        
                        # Skip ahead to avoid duplicate detections
                        i = j
                        continue
            
            i += 1
    
    def _calculate_click_severity(self, click_count: int) -> str:
        """Calculate severity based on number of repeated clicks."""
        if click_count >= 6:
            return 'high'
        elif click_count >= 4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_hesitation_severity(self, duration: float) -> str:
        """Calculate severity based on hesitation duration."""
        if duration >= 8:
            return 'high'
        elif duration >= 6:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confusion_severity(
        self,
        direction_changes: int,
        distance: float
    ) -> str:
        """Calculate severity based on navigation confusion metrics."""
        score = direction_changes * 0.5 + distance / 500
        
        if score >= 8:
            return 'high'
        elif score >= 5:
            return 'medium'
        else:
            return 'low'
    
    def _get_ui_context(
        self,
        frame_number: int,
        x: int,
        y: int
    ) -> Optional[str]:
        """Get UI element context at position."""
        elements = self._ui_elements.get(frame_number, [])
        
        for elem in elements:
            if elem.x1 <= x <= elem.x2 and elem.y1 <= y <= elem.y2:
                return f"a {elem.element_type}"
        
        return None
    
    def _is_duplicate_issue(
        self,
        issue_type: str,
        timestamp: float,
        location: tuple[int, int],
        time_threshold: float = 2.0,
        distance_threshold: int = 100
    ) -> bool:
        """Check if a similar issue was already detected."""
        for issue in self._issues:
            if issue.type != issue_type:
                continue
            
            time_diff = abs(issue.timestamp_seconds - timestamp)
            
            if time_diff > time_threshold:
                continue
            
            if issue.location:
                dist = calculate_distance(
                    location[0], location[1],
                    issue.location[0], issue.location[1]
                )
                
                if dist <= distance_threshold:
                    return True
        
        return False
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics of detected issues."""
        stats = {
            'total_issues': len(self._issues),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int)
        }
        
        for issue in self._issues:
            stats['by_type'][issue.type] += 1
            stats['by_severity'][issue.severity] += 1
        
        return stats
    
    def calculate_friction_score(self, duration_seconds: float) -> float:
        """
        Calculate overall friction score.
        
        Args:
            duration_seconds: Total video duration
            
        Returns:
            Score between 0.0 (no friction) and 1.0 (high friction)
        """
        return calculate_friction_score(self._issues, duration_seconds)

