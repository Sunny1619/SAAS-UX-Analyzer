"""
Demo data generator for testing without real video input.

Generates realistic synthetic UX analysis results.
"""

import random
from typing import Optional


def generate_demo_analysis(
    duration_seconds: float = 120.0,
    issue_count: Optional[int] = None
) -> dict:
    """
    Generate a realistic demo analysis result.
    
    Args:
        duration_seconds: Simulated video duration
        issue_count: Number of issues to generate (random if None)
        
    Returns:
        Analysis result dictionary matching API schema
    """
    if issue_count is None:
        issue_count = random.randint(3, 8)
    
    issues = []
    timestamps_used = set()
    
    issue_templates = [
        {
            'type': 'repeated_click',
            'descriptions': [
                "User clicked the login button {count} times within 5 seconds. The button may not provide adequate feedback.",
                "Multiple clicks detected on the submit form area. Consider adding loading state indicator.",
                "User rage-clicked the navigation menu {count} times. The menu may have poor responsiveness.",
                "Repeated clicks on the 'Save' button detected. Action confirmation may be unclear."
            ],
            'severity_weights': {'low': 0.2, 'medium': 0.5, 'high': 0.3}
        },
        {
            'type': 'hesitation',
            'descriptions': [
                "User hovered over the pricing section for {duration:.1f} seconds without action. Consider clarifying the value proposition.",
                "Cursor paused on the dropdown menu for {duration:.1f} seconds. Menu items may need clearer labels.",
                "User hesitated at the checkout form for {duration:.1f} seconds. Form fields may be confusing.",
                "Extended pause detected over the navigation bar. Users may be uncertain where to find features."
            ],
            'severity_weights': {'low': 0.3, 'medium': 0.4, 'high': 0.3}
        },
        {
            'type': 'navigation_confusion',
            'descriptions': [
                "User cursor moved erratically between sidebar and main content {count} times. Navigation structure may be unclear.",
                "Back-and-forth movement detected between tabs. Tab labels may not clearly indicate content.",
                "Confused cursor pattern in the settings area. Consider reorganizing settings categories.",
                "User searched multiple areas looking for a feature. Consider adding search functionality or reorganizing menu."
            ],
            'severity_weights': {'low': 0.25, 'medium': 0.45, 'high': 0.3}
        }
    ]
    
    for _ in range(issue_count):
        # Pick random issue type
        template = random.choice(issue_templates)
        
        # Generate unique timestamp
        max_attempts = 100
        for _ in range(max_attempts):
            ts_seconds = random.uniform(5, duration_seconds - 5)
            ts_rounded = round(ts_seconds)
            if ts_rounded not in timestamps_used:
                timestamps_used.add(ts_rounded)
                break
        
        # Format timestamp
        minutes = int(ts_seconds // 60)
        seconds = int(ts_seconds % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Pick severity based on weights
        severity = random.choices(
            list(template['severity_weights'].keys()),
            weights=list(template['severity_weights'].values())
        )[0]
        
        # Generate description
        description = random.choice(template['descriptions'])
        
        # Fill in template variables
        if '{count}' in description:
            count = random.randint(3, 7) if template['type'] == 'repeated_click' else random.randint(4, 8)
            description = description.format(count=count)
        if '{duration:.1f}' in description:
            duration = random.uniform(4.5, 10.0)
            description = description.format(duration=duration)
        
        issues.append({
            'type': template['type'],
            'timestamp': timestamp,
            'severity': severity,
            'description': description
        })
    
    # Sort by timestamp
    issues.sort(key=lambda x: x['timestamp'])
    
    # Calculate friction score
    severity_weights = {'low': 0.1, 'medium': 0.25, 'high': 0.4}
    weighted_score = sum(severity_weights[i['severity']] for i in issues)
    friction_score = min(1.0, weighted_score / (issue_count * 0.3))
    friction_score = round(friction_score, 3)
    
    # Count by type
    issues_by_type = {}
    for issue in issues:
        issues_by_type[issue['type']] = issues_by_type.get(issue['type'], 0) + 1
    
    # Format duration
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    total_duration = f"{minutes}m {seconds}s"
    
    # Generate heatmap data
    heatmap_data = generate_demo_heatmap()
    
    return {
        'summary': {
            'total_duration': total_duration,
            'friction_score': friction_score,
            'total_frames_analyzed': int(duration_seconds * 30 / 3),  # 30fps, skip 3
            'total_issues': len(issues),
            'issues_by_type': issues_by_type
        },
        'issues': issues,
        'heatmap_data': heatmap_data
    }


def generate_demo_heatmap(
    width: int = 1920,
    height: int = 1080,
    num_hotspots: int = 5
) -> list[dict]:
    """
    Generate demo heatmap data with realistic hotspot patterns.
    
    Args:
        width: Frame width
        height: Frame height
        num_hotspots: Number of cursor activity hotspots
        
    Returns:
        List of heatmap data points
    """
    heatmap = []
    
    # Common UI areas (normalized coordinates)
    common_areas = [
        (0.1, 0.1, "top-left navigation"),
        (0.5, 0.1, "top center"),
        (0.9, 0.1, "top-right actions"),
        (0.1, 0.5, "sidebar"),
        (0.5, 0.5, "main content center"),
        (0.7, 0.5, "main content right"),
        (0.5, 0.8, "bottom content"),
        (0.9, 0.9, "bottom-right"),
    ]
    
    # Generate hotspots
    for _ in range(num_hotspots):
        # Pick a common area with some randomization
        base_x, base_y, _ = random.choice(common_areas)
        
        # Add variation
        center_x = int((base_x + random.uniform(-0.1, 0.1)) * width)
        center_y = int((base_y + random.uniform(-0.1, 0.1)) * height)
        
        # Clamp to valid range
        center_x = max(50, min(width - 50, center_x))
        center_y = max(50, min(height - 50, center_y))
        
        # Generate points around hotspot
        intensity = random.uniform(0.6, 1.0)
        spread = random.randint(30, 100)
        
        for _ in range(random.randint(10, 30)):
            x = center_x + random.randint(-spread, spread)
            y = center_y + random.randint(-spread, spread)
            
            # Distance-based intensity falloff
            dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
            point_intensity = max(0.1, intensity * (1 - dist / (spread * 2)))
            
            heatmap.append({
                'x': max(0, min(width, x)),
                'y': max(0, min(height, y)),
                'intensity': round(point_intensity, 3),
                'count': random.randint(1, 10)
            })
    
    return heatmap


def generate_demo_cursor_events(
    duration_seconds: float = 60.0,
    fps: float = 30.0
) -> list[dict]:
    """
    Generate demo cursor movement events.
    
    Args:
        duration_seconds: Video duration
        fps: Frames per second
        
    Returns:
        List of cursor event dictionaries
    """
    events = []
    
    # Starting position
    x, y = 960, 540  # Center of 1080p screen
    
    total_frames = int(duration_seconds * fps)
    
    for frame in range(0, total_frames, 3):  # Skip every 3rd frame
        timestamp = frame / fps
        
        # Random movement with momentum
        dx = random.randint(-30, 30)
        dy = random.randint(-30, 30)
        
        x = max(0, min(1920, x + dx))
        y = max(0, min(1080, y + dy))
        
        # Random clicks (about 1% of frames)
        is_click = random.random() < 0.01
        
        events.append({
            'x': x,
            'y': y,
            'frame_number': frame,
            'timestamp_seconds': timestamp,
            'is_click': is_click,
            'confidence': random.uniform(0.7, 1.0)
        })
    
    return events

