"""
Streamlit Dashboard for UX Friction Analysis Visualization.

Provides an interactive interface to:
- Upload videos for analysis
- View analysis results
- Explore detected issues on a timeline
- Visualize cursor activity heatmap
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import json

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="SaaS UX Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom Styling
# ============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #475569;
        margin-bottom: 16px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Severity badges */
    .severity-high {
        background-color: #fecaca;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .severity-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .severity-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Issue type badges */
    .issue-repeated_click {
        background-color: #fee2e2;
        color: #dc2626;
    }
    
    .issue-hesitation {
        background-color: #fef3c7;
        color: #d97706;
    }
    
    .issue-navigation_confusion {
        background-color: #dbeafe;
        color: #2563eb;
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
    }
    
    /* Timeline item */
    .timeline-item {
        border-left: 3px solid #6366f1;
        padding-left: 20px;
        margin-bottom: 20px;
        position: relative;
    }
    
    .timeline-item::before {
        content: '';
        width: 12px;
        height: 12px;
        background: #6366f1;
        border-radius: 50%;
        position: absolute;
        left: -7.5px;
        top: 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_health() -> tuple[bool, dict]:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, {}
    except Exception:
        return False, {}


def analyze_video(file) -> Optional[dict]:
    """Send video to API for analysis."""
    try:
        files = {"file": (file.name, file.getvalue(), "video/mp4")}
        response = requests.post(f"{API_BASE_URL}/analyze", files=files, timeout=300)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def get_demo_analysis() -> Optional[dict]:
    """Get demo analysis from API."""
    try:
        response = requests.post(f"{API_BASE_URL}/analyze/demo", timeout=30)
        if response.status_code == 200:
            return response.json()
        raise Exception("API not available")
    except Exception:
        # Fallback to local demo generation
        try:
            from demo.generator import generate_demo_analysis
        except ImportError:
            sys.path.insert(0, str(PROJECT_ROOT))
            from demo.generator import generate_demo_analysis
        return generate_demo_analysis()


def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        'high': '#ef4444',
        'medium': '#f59e0b',
        'low': '#10b981'
    }
    return colors.get(severity, '#6b7280')


def get_issue_type_color(issue_type: str) -> str:
    """Get color for issue type."""
    colors = {
        'repeated_click': '#ef4444',
        'hesitation': '#f59e0b',
        'navigation_confusion': '#3b82f6'
    }
    return colors.get(issue_type, '#6b7280')


def get_issue_type_icon(issue_type: str) -> str:
    """Get icon for issue type."""
    icons = {
        'repeated_click': '🖱️',
        'hesitation': '⏸️',
        'navigation_confusion': '🔀'
    }
    return icons.get(issue_type, '❓')


# ============================================================================
# Dashboard Components
# ============================================================================

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 SaaS UX Analyzer</h1>
        <p>AI-powered detection of UX friction points in screen recordings</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(summary: dict):
    """Render key metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Duration",
            value=summary['total_duration']
        )
    
    with col2:
        friction_score = summary['friction_score']
        delta_color = "inverse" if friction_score > 0.5 else "normal"
        st.metric(
            label="Friction Score",
            value=f"{friction_score:.1%}",
            delta="High friction" if friction_score > 0.5 else "Low friction",
            delta_color=delta_color
        )
    
    with col3:
        st.metric(
            label="Issues Detected",
            value=summary.get('total_issues', 0)
        )
    
    with col4:
        st.metric(
            label="Frames Analyzed",
            value=f"{summary.get('total_frames_analyzed', 0):,}"
        )


def render_friction_gauge(score: float):
    """Render friction score gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Friction Score", 'font': {'size': 20, 'color': '#f8fafc'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#475569'},
            'bar': {'color': '#6366f1'},
            'bgcolor': '#1e293b',
            'borderwidth': 2,
            'bordercolor': '#475569',
            'steps': [
                {'range': [0, 30], 'color': '#10b981'},
                {'range': [30, 60], 'color': '#f59e0b'},
                {'range': [60, 100], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': '#f8fafc', 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f8fafc'},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig)


def render_issues_by_type(issues: list):
    """Render bar chart of issues by type."""
    if not issues:
        st.info("No issues detected")
        return
    
    type_counts = {}
    for issue in issues:
        t = issue['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    df = pd.DataFrame([
        {'Type': k.replace('_', ' ').title(), 'Count': v, 'Raw': k}
        for k, v in type_counts.items()
    ])
    
    colors = [get_issue_type_color(t) for t in df['Raw']]
    
    fig = px.bar(
        df,
        x='Type',
        y='Count',
        color='Type',
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f8fafc'},
        showlegend=False,
        xaxis={'title': '', 'gridcolor': '#334155'},
        yaxis={'title': 'Count', 'gridcolor': '#334155'},
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig)


def render_severity_distribution(issues: list):
    """Render pie chart of severity distribution."""
    if not issues:
        return
    
    severity_counts = {}
    for issue in issues:
        s = issue['severity']
        severity_counts[s] = severity_counts.get(s, 0) + 1
    
    df = pd.DataFrame([
        {'Severity': k.title(), 'Count': v, 'Raw': k}
        for k, v in severity_counts.items()
    ])
    
    colors = [get_severity_color(s) for s in df['Raw']]
    
    fig = px.pie(
        df,
        values='Count',
        names='Severity',
        color_discrete_sequence=colors,
        hole=0.4
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f8fafc'},
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig)


def render_timeline(issues: list):
    """Render timeline of issues."""
    st.subheader("📅 Issue Timeline")
    
    if not issues:
        st.info("No issues to display")
        return
    
    for issue in issues:
        icon = get_issue_type_icon(issue['type'])
        severity_color = get_severity_color(issue['severity'])
        
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1e293b, #334155);
                    border-radius: 8px;
                    padding: 12px;
                    text-align: center;
                    border: 1px solid #475569;
                ">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">
                        {issue['timestamp']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1e293b, #334155);
                    border-radius: 8px;
                    padding: 16px;
                    border-left: 4px solid {severity_color};
                    border: 1px solid #475569;
                ">
                    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                        <span style="
                            background-color: {severity_color}20;
                            color: {severity_color};
                            padding: 4px 12px;
                            border-radius: 9999px;
                            font-size: 0.75rem;
                            font-weight: 600;
                        ">{issue['severity'].upper()}</span>
                        <span style="
                            background-color: #475569;
                            color: #f8fafc;
                            padding: 4px 12px;
                            border-radius: 9999px;
                            font-size: 0.75rem;
                        ">{issue['type'].replace('_', ' ').title()}</span>
                    </div>
                    <p style="color: #cbd5e1; margin: 0; line-height: 1.6;">
                        {issue['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)


def render_heatmap(heatmap_data: list, width: int = 1920, height: int = 1080):
    """Render cursor activity heatmap."""
    st.subheader("🔥 Cursor Activity Heatmap")
    
    if not heatmap_data:
        st.info("No heatmap data available")
        return
    
    # Create heatmap visualization
    df = pd.DataFrame(heatmap_data)
    
    fig = go.Figure()
    
    # Add scatter points with size based on intensity
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=df['intensity'] * 30 + 5,
            color=df['intensity'],
            colorscale='Hot',
            opacity=0.6,
            showscale=True,
            colorbar=dict(
                title=dict(text='Activity', font=dict(color='#f8fafc')),
                tickfont=dict(color='#f8fafc')
            )
        ),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{marker.color:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1e293b',
        font={'color': '#f8fafc'},
        xaxis={
            'title': 'X Position',
            'range': [0, width],
            'gridcolor': '#334155',
            'zeroline': False
        },
        yaxis={
            'title': 'Y Position',
            'range': [height, 0],  # Flip Y axis
            'gridcolor': '#334155',
            'zeroline': False
        },
        height=500,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    st.plotly_chart(fig)


def render_issues_table(issues: list):
    """Render issues in a table format."""
    st.subheader("📋 All Issues")
    
    if not issues:
        st.info("No issues detected")
        return
    
    df = pd.DataFrame(issues)
    df['type'] = df['type'].apply(lambda x: x.replace('_', ' ').title())
    df['severity'] = df['severity'].apply(lambda x: x.title())
    df.columns = ['Type', 'Timestamp', 'Severity', 'Description']
    
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            'Type': st.column_config.TextColumn('Issue Type', width='medium'),
            'Timestamp': st.column_config.TextColumn('Time', width='small'),
            'Severity': st.column_config.TextColumn('Severity', width='small'),
            'Description': st.column_config.TextColumn('Description', width='large')
        }
    )


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Analysis Mode")
        
        # Check API status
        api_available, health_info = check_api_health()
        
        if api_available:
            st.success("✅ API Connected")
            if health_info.get('yolo_available'):
                st.info("🤖 YOLOv8 Model Active")
            else:
                st.warning("⚠️ Using Heuristic Detection")
        else:
            st.warning("⚠️ API Offline - Demo Mode Only")
        
        st.markdown("---")
        
        analysis_mode = st.radio(
            "Select Mode",
            ["Upload Video", "Demo Analysis"],
            index=1 if not api_available else 0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool analyzes screen recordings to detect UX friction points:
        
        - **🖱️ Repeated Clicks**: Rage clicks or unclear UI
        - **⏸️ Hesitation**: Confusion or unclear CTAs  
        - **🔀 Navigation Confusion**: Erratic cursor patterns
        """)
    
    # Main content
    if analysis_mode == "Upload Video":
        st.markdown("### 📤 Upload Video for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a screen recording (MP4, AVI, MOV)",
            type=['mp4', 'avi', 'mov', 'webm'],
            help="Upload a video of a user session to analyze"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.video(uploaded_file)
            
            with col2:
                st.markdown(f"""
                **File Details:**
                - Name: `{uploaded_file.name}`
                - Size: {uploaded_file.size / (1024*1024):.2f} MB
                """)
                
                if st.button("🔍 Analyze Video", type="primary"):
                    with st.spinner("Analyzing video... This may take a moment."):
                        result = analyze_video(uploaded_file)
                        
                        if result:
                            st.session_state['analysis_result'] = result
                            st.success("✅ Analysis complete!")
                            st.rerun()
    
    else:  # Demo Analysis
        st.markdown("### 🎮 Demo Analysis")
        st.info("Click below to see a sample analysis with synthetic data")
        
        if st.button("🚀 Run Demo Analysis", type="primary"):
            with st.spinner("Generating demo analysis..."):
                result = get_demo_analysis()
                
                if result:
                    st.session_state['analysis_result'] = result
                    st.success("✅ Demo analysis generated!")
                    st.rerun()
    
    # Display results if available
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Metrics
        render_metrics(result['summary'])
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            render_friction_gauge(result['summary']['friction_score'])
        
        with col2:
            render_issues_by_type(result['issues'])
        
        # Second row
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("📊 Severity Distribution")
            render_severity_distribution(result['issues'])
        
        with col4:
            st.subheader("📈 Issues Summary")
            issues_by_type = result['summary'].get('issues_by_type', {})
            for issue_type, count in issues_by_type.items():
                icon = get_issue_type_icon(issue_type)
                st.markdown(f"{icon} **{issue_type.replace('_', ' ').title()}**: {count}")
        
        # Heatmap
        if result.get('heatmap_data'):
            render_heatmap(result['heatmap_data'])
        
        # Timeline
        render_timeline(result['issues'])
        
        # Full table
        render_issues_table(result['issues'])
        
        # Export option
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        export_data = json.dumps(result, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=export_data,
            file_name="ux_analysis_report.json",
            mime="application/json"
        )
        
        # Clear results button
        if st.button("🔄 Clear Results"):
            del st.session_state['analysis_result']
            st.rerun()


if __name__ == "__main__":
    main()

