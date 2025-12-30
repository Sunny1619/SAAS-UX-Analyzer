#!/usr/bin/env python
"""
Quick start script for the UX Analyzer.

Usage:
    python run.py api        # Start the FastAPI backend
    python run.py dashboard  # Start the Streamlit dashboard
    python run.py both       # Start both (API in background)
    python run.py demo       # Run a demo analysis
"""

import sys
import subprocess
import time
import requests


def start_api():
    """Start the FastAPI server."""
    print("🚀 Starting UX Analyzer API...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.api:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ])


def start_dashboard():
    """Start the Streamlit dashboard."""
    print("🎨 Starting UX Analyzer Dashboard...")
    print("   URL: http://localhost:8501")
    print("")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.address", "localhost",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])


def start_both():
    """Start both API and dashboard."""
    import threading
    
    print("🚀 Starting UX Analyzer (API + Dashboard)...")
    print("")
    
    # Start API in background thread
    api_thread = threading.Thread(target=lambda: subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.api:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]), daemon=True)
    api_thread.start()
    
    # Wait for API to be ready
    print("⏳ Waiting for API to start...")
    for _ in range(30):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except Exception:
            time.sleep(1)
    
    print("")
    # Start dashboard in foreground
    start_dashboard()


def run_demo():
    """Run a demo analysis and print results."""
    print("🎮 Running Demo Analysis...")
    print("")
    
    try:
        # Try API first
        response = requests.post("http://localhost:8000/analyze/demo", timeout=10)
        if response.status_code == 200:
            result = response.json()
        else:
            raise Exception("API not available")
    except Exception:
        # Fall back to local generation
        print("⚠️  API not running, using local demo generator...")
        from demo.generator import generate_demo_analysis
        result = generate_demo_analysis()
    
    # Print results
    summary = result['summary']
    print("=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Duration:        {summary['total_duration']}")
    print(f"  Friction Score:  {summary['friction_score']:.1%}")
    print(f"  Total Issues:    {summary.get('total_issues', len(result['issues']))}")
    print("")
    
    print("📋 DETECTED ISSUES")
    print("-" * 60)
    
    for i, issue in enumerate(result['issues'], 1):
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue['severity'], "⚪")
        type_icon = {
            "repeated_click": "🖱️",
            "hesitation": "⏸️",
            "navigation_confusion": "🔀"
        }.get(issue['type'], "❓")
        
        print(f"\n{i}. [{issue['timestamp']}] {type_icon} {issue['type'].replace('_', ' ').title()}")
        print(f"   {severity_icon} Severity: {issue['severity'].upper()}")
        print(f"   📝 {issue['description']}")
    
    print("")
    print("=" * 60)
    print("✅ Demo analysis complete!")


def print_usage():
    """Print usage information."""
    print(__doc__)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "api":
        start_api()
    elif command == "dashboard":
        start_dashboard()
    elif command == "both":
        start_both()
    elif command == "demo":
        run_demo()
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()

