"""
Utility functions for the UX analyzer.
"""

import math
from typing import Optional
from collections import defaultdict


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to mm:ss format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string in "mm:ss" format
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    Convert seconds to "Xm Ys" format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string in "Xm Ys" format
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        
    Returns:
        Euclidean distance
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def is_within_region(
    x: int, 
    y: int, 
    region_x: int, 
    region_y: int, 
    threshold: int = 50
) -> bool:
    """
    Check if a point is within a threshold distance of a region center.
    
    Args:
        x, y: Point coordinates
        region_x, region_y: Region center coordinates
        threshold: Maximum distance to be considered "within" region
        
    Returns:
        True if point is within region
    """
    return calculate_distance(x, y, region_x, region_y) <= threshold


def calculate_velocity(
    positions: list[tuple[int, int, float]]
) -> list[float]:
    """
    Calculate cursor velocity between consecutive positions.
    
    Args:
        positions: List of (x, y, timestamp) tuples
        
    Returns:
        List of velocities (pixels per second)
    """
    velocities = []
    for i in range(1, len(positions)):
        x1, y1, t1 = positions[i - 1]
        x2, y2, t2 = positions[i]
        
        distance = calculate_distance(x1, y1, x2, y2)
        time_delta = t2 - t1
        
        if time_delta > 0:
            velocities.append(distance / time_delta)
        else:
            velocities.append(0.0)
    
    return velocities


def generate_heatmap_data(
    cursor_positions: list[dict],
    grid_size: int = 20,
    width: int = 1920,
    height: int = 1080
) -> list[dict]:
    """
    Generate heatmap data from cursor positions.
    
    Args:
        cursor_positions: List of cursor position dicts with 'x' and 'y'
        grid_size: Size of grid cells for aggregation
        width: Video frame width
        height: Video frame height
        
    Returns:
        List of heatmap cells with x, y, and intensity
    """
    grid = defaultdict(int)
    
    for pos in cursor_positions:
        if pos.get('x') is not None and pos.get('y') is not None:
            grid_x = pos['x'] // grid_size
            grid_y = pos['y'] // grid_size
            grid[(grid_x, grid_y)] += 1
    
    if not grid:
        return []
    
    max_count = max(grid.values())
    
    heatmap = []
    for (grid_x, grid_y), count in grid.items():
        heatmap.append({
            'x': grid_x * grid_size + grid_size // 2,
            'y': grid_y * grid_size + grid_size // 2,
            'intensity': count / max_count if max_count > 0 else 0,
            'count': count
        })
    
    return heatmap


def detect_direction_changes(
    positions: list[tuple[int, int]]
) -> int:
    """
    Count significant direction changes in cursor movement.
    
    Args:
        positions: List of (x, y) tuples
        
    Returns:
        Number of direction changes
    """
    if len(positions) < 3:
        return 0
    
    direction_changes = 0
    
    for i in range(1, len(positions) - 1):
        # Calculate vectors
        dx1 = positions[i][0] - positions[i - 1][0]
        dy1 = positions[i][1] - positions[i - 1][1]
        dx2 = positions[i + 1][0] - positions[i][0]
        dy2 = positions[i + 1][1] - positions[i][1]
        
        # Calculate angle between vectors using dot product
        dot = dx1 * dx2 + dy1 * dy2
        mag1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        mag2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot / (mag1 * mag2)
            # Clamp to valid range for acos
            cos_angle = max(-1, min(1, cos_angle))
            angle = math.acos(cos_angle)
            
            # Count as direction change if angle > 90 degrees
            if angle > math.pi / 2:
                direction_changes += 1
    
    return direction_changes


def cluster_positions(
    positions: list[tuple[int, int, float]],
    distance_threshold: int = 50
) -> list[list[tuple[int, int, float]]]:
    """
    Cluster cursor positions that are close together.
    
    Args:
        positions: List of (x, y, timestamp) tuples
        distance_threshold: Max distance to be in same cluster
        
    Returns:
        List of position clusters
    """
    if not positions:
        return []
    
    clusters = []
    current_cluster = [positions[0]]
    
    for pos in positions[1:]:
        last_pos = current_cluster[-1]
        dist = calculate_distance(pos[0], pos[1], last_pos[0], last_pos[1])
        
        if dist <= distance_threshold:
            current_cluster.append(pos)
        else:
            if len(current_cluster) > 0:
                clusters.append(current_cluster)
            current_cluster = [pos]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def calculate_friction_score(
    issues: list,
    duration_seconds: float,
    weights: Optional[dict] = None
) -> float:
    """
    Calculate overall friction score based on detected issues.
    
    Args:
        issues: List of UXIssue objects
        duration_seconds: Total video duration
        weights: Optional custom weights for issue types
        
    Returns:
        Friction score between 0.0 and 1.0
    """
    if not issues or duration_seconds <= 0:
        return 0.0
    
    default_weights = {
        'repeated_click': {'low': 0.1, 'medium': 0.2, 'high': 0.3},
        'hesitation': {'low': 0.05, 'medium': 0.1, 'high': 0.2},
        'navigation_confusion': {'low': 0.15, 'medium': 0.25, 'high': 0.35}
    }
    
    weights = weights or default_weights
    
    total_score = 0.0
    for issue in issues:
        issue_type = issue.type if hasattr(issue, 'type') else issue.get('type')
        severity = issue.severity if hasattr(issue, 'severity') else issue.get('severity')
        
        # Handle enum values
        if hasattr(issue_type, 'value'):
            issue_type = issue_type.value
        if hasattr(severity, 'value'):
            severity = severity.value
        
        if issue_type in weights and severity in weights[issue_type]:
            total_score += weights[issue_type][severity]
    
    # Normalize by duration (more issues per minute = higher score)
    issues_per_minute = len(issues) / (duration_seconds / 60)
    duration_factor = min(1.0, issues_per_minute / 10)  # Cap at 10 issues/min
    
    # Combine weighted score with frequency
    friction_score = min(1.0, (total_score * 0.7) + (duration_factor * 0.3))
    
    return round(friction_score, 3)

