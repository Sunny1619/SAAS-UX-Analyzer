# SaaS UX Analyzer 🔍

An AI-powered computer vision system that analyzes SaaS user screen-recording videos to automatically detect UX friction points.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)

## 🎯 Problem Statement

SaaS companies need to understand where users struggle in their products. Manual review of screen recordings is time-consuming and inconsistent. This project provides an automated solution that:

- **Detects UX friction patterns** using computer vision and rule-based ML
- **Quantifies user frustration** with an explainable friction score
- **Generates actionable insights** with timestamps and descriptions
- **Scales to handle multiple videos** through a clean API design

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SaaS UX Analyzer                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐  │
│  │   Frontend   │    │                  Backend (FastAPI)                │  │
│  │  (Streamlit) │───▶│  POST /analyze                                   │  │
│  │              │    │       │                                          │  │
│  │  • Upload    │    │       ▼                                          │  │
│  │  • Timeline  │    │  ┌────────────────┐    ┌──────────────────┐     │  │
│  │  • Heatmap   │    │  │ Video Processor │───▶│  Cursor Tracker  │     │  │
│  │  • Charts    │    │  │    (OpenCV)    │    │ (Motion Detect)  │     │  │
│  └──────────────┘    │  └────────────────┘    └──────────────────┘     │  │
│                      │          │                      │                │  │
│                      │          ▼                      ▼                │  │
│                      │  ┌────────────────┐    ┌──────────────────┐     │  │
│                      │  │  UI Detector   │    │   UX Analyzer    │     │  │
│                      │  │   (YOLOv8)     │───▶│  (Rule-based)    │     │  │
│                      │  └────────────────┘    └──────────────────┘     │  │
│                      │                                │                 │  │
│                      │                                ▼                 │  │
│                      │                       ┌──────────────────┐       │  │
│                      │                       │  Analysis Report │       │  │
│                      │                       │  • Friction Score│       │  │
│                      │                       │  • Issue List    │       │  │
│                      │                       │  • Heatmap Data  │       │  │
│                      │                       └──────────────────┘       │  │
│                      └──────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Storage Layer (S3/Local)                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔬 How It Works

### Step 1: Video Processing
- Load MP4 video using OpenCV
- Extract frames with configurable skip rate (default: every 3rd frame)
- Resize frames for optimal processing speed

### Step 2: Cursor Tracking
- Use frame differencing to detect motion
- Track cursor position through contour analysis
- Detect clicks based on cursor stationarity patterns

### Step 3: UI Element Detection
- YOLOv8 model detects UI elements (buttons, inputs, menus)
- Fallback heuristic detection using edge analysis
- Provides context for friction analysis

### Step 4: Friction Detection (Rule-Based, Explainable)

| Issue Type | Detection Rule | Severity Criteria |
|------------|---------------|-------------------|
| **Repeated Clicks** | 3+ clicks in 50px radius within 5 seconds | Low: 3-4, Medium: 4-6, High: 6+ |
| **Hesitation** | Cursor stationary for 4+ seconds without click | Low: 4-6s, Medium: 6-8s, High: 8s+ |
| **Navigation Confusion** | 6+ direction changes within 3 seconds | Based on changes × distance |

### Step 5: Report Generation
- Calculate friction score (0.0 - 1.0)
- Generate timestamped issue list
- Create cursor activity heatmap

## 📁 Project Structure

```
saas-ux-analyzer/
├── app/
│   ├── __init__.py
│   ├── api.py                 # FastAPI backend
│   ├── services/
│   │   ├── __init__.py
│   │   ├── video_processor.py # OpenCV video handling
│   │   ├── cursor_tracker.py  # Motion-based cursor tracking
│   │   ├── ui_detector.py     # YOLOv8 UI detection
│   │   └── ux_analyzer.py     # Friction detection logic
│   └── models/
│       ├── __init__.py
│       └── schemas.py         # Pydantic models
├── dashboard/
│   ├── __init__.py
│   └── app.py                 # Streamlit dashboard
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Utility functions
├── demo/
│   ├── __init__.py
│   └── generator.py           # Demo data generation
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd saas-ux-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Start the FastAPI server
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000

### Running the Dashboard

```bash
# In a new terminal
streamlit run dashboard/app.py
```

Dashboard will be available at: http://localhost:8501

## 📡 API Reference

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "yolo_available": true
}
```

### Analyze Video
```http
POST /analyze
Content-Type: multipart/form-data
```

**Request Body:** MP4 video file

**Response:**
```json
{
  "summary": {
    "total_duration": "2m 0s",
    "friction_score": 0.42,
    "total_frames_analyzed": 1200,
    "total_issues": 5,
    "issues_by_type": {
      "repeated_click": 2,
      "hesitation": 2,
      "navigation_confusion": 1
    }
  },
  "issues": [
    {
      "type": "repeated_click",
      "timestamp": "00:23",
      "severity": "medium",
      "description": "User clicked the same area 4 times within 5 seconds near a button. This may indicate unclear UI or unresponsive element."
    },
    {
      "type": "hesitation",
      "timestamp": "00:45",
      "severity": "low",
      "description": "User hesitated for 5.2 seconds without interaction while hovering over a menu. This may indicate confusion or unclear call-to-action."
    }
  ],
  "heatmap_data": [
    {"x": 450, "y": 300, "intensity": 0.85, "count": 15},
    {"x": 800, "y": 200, "intensity": 0.62, "count": 8}
  ]
}
```

### Demo Analysis
```http
POST /analyze/demo
```

Returns a synthetic analysis result for testing.

## ☁️ AWS Deployment

### Architecture for Production

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Route 53  │────▶│     ALB     │────▶│    ECS/EC2  │       │
│   │   (DNS)     │     │             │     │   (API)     │       │
│   └─────────────┘     └─────────────┘     └──────┬──────┘       │
│                                                   │              │
│                           ┌───────────────────────┤              │
│                           │                       │              │
│                           ▼                       ▼              │
│                    ┌─────────────┐         ┌─────────────┐       │
│                    │     S3      │         │  CloudWatch │       │
│                    │  (Videos)   │         │   (Logs)    │       │
│                    └─────────────┘         └─────────────┘       │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Streamlit on ECS                      │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Steps

1. **Containerize the Application**
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Push to ECR**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t ux-analyzer .
docker tag ux-analyzer:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ux-analyzer:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ux-analyzer:latest
```

3. **Configure S3 Storage**
```python
# Update StorageAdapter in api.py
storage = StorageAdapter(
    storage_type="s3",
    bucket="your-video-bucket"
)
```

4. **Environment Variables**
```bash
AWS_REGION=us-east-1
S3_BUCKET=ux-analyzer-videos
MODEL_PATH=s3://models/yolov8n.pt
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Test the API manually
curl -X POST "http://localhost:8000/analyze/demo"

# Test with a real video
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@sample_recording.mp4"
```

## 📊 Demo Mode

The system includes a demo mode for testing without real videos:

1. Start the API server
2. Open the dashboard
3. Click "Demo Analysis" 
4. View synthetic results with realistic friction patterns

## 🔮 Future Improvements

### Short-term
- [ ] Add more UI element classes to YOLO model
- [ ] Implement click sound detection for more accurate click identification
- [ ] Add session comparison features
- [ ] Export reports to PDF

### Medium-term
- [ ] Train custom YOLO model on UI element dataset
- [ ] Add ML-based click prediction (beyond heuristics)
- [ ] Implement real-time streaming analysis
- [ ] Add A/B test comparison mode

### Long-term
- [ ] Integrate eye-tracking data
- [ ] Add NLP analysis of on-screen text
- [ ] Build recommendation engine for UX improvements
- [ ] Create browser extension for live analysis

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Video Processing | OpenCV 4.8+ |
| Object Detection | YOLOv8 (Ultralytics) |
| Backend API | FastAPI |
| Frontend Dashboard | Streamlit |
| Data Validation | Pydantic |
| Visualization | Plotly |
| Cloud Storage | AWS S3 (abstracted) |
| Deployment | Docker, AWS ECS |

## 📄 License
Built as a demonstration of computer vision, ML engineering, and production-ready Python development skills.




