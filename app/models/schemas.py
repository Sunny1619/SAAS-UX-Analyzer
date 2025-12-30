"""
Pydantic models for data validation and serialization.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IssueType(str, Enum):
    """Types of UX friction issues detected."""
    REPEATED_CLICK = "repeated_click"
    HESITATION = "hesitation"
    NAVIGATION_CONFUSION = "navigation_confusion"


class Severity(str, Enum):
    """Severity levels for detected issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CursorPosition(BaseModel):
    """Represents cursor position at a specific frame."""
    x: int
    y: int
    frame_number: int
    timestamp_seconds: float
    is_click: bool = False


class UIElement(BaseModel):
    """Detected UI element from YOLOv8."""
    element_type: str  # button, input, menu, etc.
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    frame_number: int


class FrameData(BaseModel):
    """Aggregated data for a single frame."""
    frame_number: int
    timestamp_seconds: float
    cursor_position: Optional[CursorPosition] = None
    ui_elements: list[UIElement] = Field(default_factory=list)


class UXIssue(BaseModel):
    """A detected UX friction issue."""
    type: IssueType
    timestamp: str  # Format: "mm:ss"
    severity: Severity
    description: str
    frame_range: tuple[int, int] = Field(default=(0, 0))
    location: Optional[tuple[int, int]] = None  # (x, y) center of issue area


class AnalysisSummary(BaseModel):
    """Summary statistics for the analysis."""
    total_duration: str  # Format: "Xm Ys"
    total_duration_seconds: float
    friction_score: float = Field(ge=0.0, le=1.0)
    total_frames_analyzed: int
    frames_per_second: float
    total_cursor_movements: int
    total_clicks_detected: int


class AnalysisResult(BaseModel):
    """Complete analysis result returned by the API."""
    summary: AnalysisSummary
    issues: list[UXIssue]
    heatmap_data: Optional[list[dict]] = None  # For visualization


class VideoMetadata(BaseModel):
    """Metadata extracted from video file."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    file_size_mb: Optional[float] = None

