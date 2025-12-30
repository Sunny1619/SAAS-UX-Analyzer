"""
FastAPI backend for UX Friction Analysis.

Provides REST API endpoints for video analysis.
"""

import os
import sys
import tempfile
import uuid
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.video_processor import VideoProcessor
from app.services.cursor_tracker import CursorTracker
from app.services.ui_detector import UIDetector
from app.services.ux_analyzer import UXAnalyzer
from utils.helpers import format_duration, generate_heatmap_data, calculate_friction_score


# ============================================================================
# Data Models for API
# ============================================================================

class UXIssueResponse(BaseModel):
    """UX issue in API response format."""
    type: str
    timestamp: str
    severity: str
    description: str


class AnalysisSummary(BaseModel):
    """Summary of the analysis."""
    total_duration: str
    friction_score: float = Field(ge=0.0, le=1.0)
    total_frames_analyzed: int = 0
    total_issues: int = 0
    issues_by_type: dict = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    summary: AnalysisSummary
    issues: list[UXIssueResponse]
    heatmap_data: Optional[list[dict]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    yolo_available: bool


# ============================================================================
# Storage Adapter (S3-compatible abstraction)
# ============================================================================

class StorageAdapter:
    """
    Abstract storage adapter for video files.
    
    Supports local filesystem and can be extended for S3.
    In production, swap implementation for actual S3 client.
    """
    
    def __init__(self, storage_type: str = "local", bucket: Optional[str] = None):
        """
        Initialize storage adapter.
        
        Args:
            storage_type: "local" or "s3"
            bucket: S3 bucket name (required for S3)
        """
        self.storage_type = storage_type
        self.bucket = bucket
        self.local_path = Path(tempfile.gettempdir()) / "ux_analyzer"
        self.local_path.mkdir(exist_ok=True)
    
    async def save_file(self, file: UploadFile, file_id: str) -> str:
        """
        Save uploaded file.
        
        Args:
            file: Uploaded file
            file_id: Unique file identifier
            
        Returns:
            Path to saved file
        """
        if self.storage_type == "local":
            file_path = self.local_path / f"{file_id}.mp4"
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            return str(file_path)
        
        elif self.storage_type == "s3":
            # S3 implementation would go here
            # Example:
            # import boto3
            # s3 = boto3.client('s3')
            # s3.upload_fileobj(file.file, self.bucket, f"{file_id}.mp4")
            # return f"s3://{self.bucket}/{file_id}.mp4"
            raise NotImplementedError("S3 storage not implemented in demo mode")
        
        raise ValueError(f"Unknown storage type: {self.storage_type}")
    
    def get_file_path(self, file_id: str) -> str:
        """Get path to stored file."""
        if self.storage_type == "local":
            return str(self.local_path / f"{file_id}.mp4")
        return f"s3://{self.bucket}/{file_id}.mp4"
    
    def delete_file(self, file_id: str):
        """Delete stored file."""
        if self.storage_type == "local":
            file_path = self.local_path / f"{file_id}.mp4"
            if file_path.exists():
                file_path.unlink()


# ============================================================================
# Analysis Service
# ============================================================================

class AnalysisService:
    """
    Orchestrates the complete video analysis pipeline.
    """
    
    def __init__(self):
        """Initialize analysis service with all components."""
        self.video_processor = VideoProcessor(frame_skip=3)
        self.cursor_tracker = CursorTracker()
        self.ui_detector = UIDetector(use_heuristics=True)
        self.ux_analyzer = UXAnalyzer()
    
    def analyze_video(self, video_path: str) -> AnalysisResponse:
        """
        Run complete analysis pipeline on a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Complete analysis response
        """
        # Load video
        metadata = self.video_processor.load_video(video_path)
        
        # Process frames
        cursor_events = []
        ui_elements = {}
        frame_count = 0
        
        # Reset tracker for new video
        self.cursor_tracker.reset()
        
        for frame_num, timestamp, frame in self.video_processor.extract_frames():
            # Track cursor
            cursor_event = self.cursor_tracker.process_frame(
                frame, frame_num, timestamp
            )
            
            if cursor_event:
                cursor_events.append(cursor_event)
            
            # Detect UI elements (every 10th processed frame for performance)
            if frame_count % 10 == 0:
                elements = self.ui_detector.detect(frame, frame_num)
                if elements:
                    ui_elements[frame_num] = elements
            
            frame_count += 1
        
        # Release video
        self.video_processor.release()
        
        # Analyze for UX issues
        issues = self.ux_analyzer.analyze(
            cursor_events=[{
                'x': e.x,
                'y': e.y,
                'frame_number': e.frame_number,
                'timestamp_seconds': e.timestamp_seconds,
                'is_click': e.is_click
            } for e in cursor_events],
            video_duration=metadata.duration_seconds
        )
        
        # Convert issues for response
        issue_responses = []
        for issue in issues:
            issue_responses.append(UXIssueResponse(
                type=issue.type,
                timestamp=issue.timestamp,
                severity=issue.severity,
                description=issue.description
            ))
        
        # Generate heatmap data
        positions = self.cursor_tracker.get_positions_as_dict()
        heatmap = generate_heatmap_data(
            positions,
            width=metadata.width,
            height=metadata.height
        )
        
        # Calculate friction score
        friction_score = self.ux_analyzer.calculate_friction_score(
            metadata.duration_seconds
        )
        
        # Build summary
        stats = self.ux_analyzer.get_summary_stats()
        summary = AnalysisSummary(
            total_duration=format_duration(metadata.duration_seconds),
            friction_score=friction_score,
            total_frames_analyzed=frame_count,
            total_issues=len(issues),
            issues_by_type=dict(stats['by_type'])
        )
        
        return AnalysisResponse(
            summary=summary,
            issues=issue_responses,
            heatmap_data=heatmap
        )


# ============================================================================
# FastAPI Application
# ============================================================================

# Global instances
storage = StorageAdapter(storage_type="local")
analysis_service = AnalysisService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("UX Analyzer API starting up...")
    yield
    # Shutdown
    print("UX Analyzer API shutting down...")


app = FastAPI(
    title="SaaS UX Analyzer API",
    description="AI-powered computer vision system for detecting UX friction points in screen recordings",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and availability of ML models.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        yolo_available=analysis_service.ui_detector.is_yolo_available
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze a screen recording video for UX friction points.
    
    **Input**: MP4 video file of a SaaS user session
    
    **Output**: Structured analysis with:
    - Summary statistics and friction score
    - List of detected UX issues with timestamps
    - Cursor heatmap data for visualization
    
    **Detected Issue Types**:
    - `repeated_click`: User clicks same area multiple times
    - `hesitation`: Cursor hovers without action
    - `navigation_confusion`: Erratic cursor movement
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported formats: MP4, AVI, MOV, WEBM"
        )
    
    # Generate unique ID for this analysis
    file_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        video_path = await storage.save_file(file, file_id)
        
        # Run analysis
        result = analysis_service.analyze_video(video_path)
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(storage.delete_file, file_id)
        else:
            storage.delete_file(file_id)
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Cleanup on error
        storage.delete_file(file_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/demo", response_model=AnalysisResponse)
async def analyze_demo():
    """
    Run analysis on demo/synthetic data.
    
    Useful for testing the API without uploading a real video.
    Returns a realistic sample analysis result.
    """
    # Import relative to project structure
    try:
        from demo.generator import generate_demo_analysis
    except ImportError:
        # Fallback for when running from different directory
        sys.path.insert(0, str(PROJECT_ROOT))
        from demo.generator import generate_demo_analysis
    
    return generate_demo_analysis()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SaaS UX Analyzer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "analyze": "POST /analyze - Upload video for analysis",
            "demo": "POST /analyze/demo - Run demo analysis"
        }
    }


# ============================================================================
# Run Configuration
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

