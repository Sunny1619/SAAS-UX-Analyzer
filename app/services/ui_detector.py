"""
UI Element Detection service using YOLOv8.

Detects common UI elements like buttons, input fields, and menus.
Designed to be modular and replaceable with other detection models.
"""

import cv2
import numpy as np
from typing import Optional, Protocol
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectedElement:
    """Represents a detected UI element."""
    element_type: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    frame_number: int = 0
    
    @property
    def center(self) -> tuple[int, int]:
        """Get center point of element."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self) -> int:
        """Get area of element bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


class DetectorProtocol(Protocol):
    """Protocol for UI element detectors (enables easy swapping)."""
    
    def detect(self, frame: np.ndarray) -> list[DetectedElement]:
        """Detect UI elements in frame."""
        ...


class UIDetector:
    """
    UI Element detector using YOLOv8.
    
    Uses a pretrained YOLOv8 model to detect common UI elements.
    Falls back to heuristic detection if YOLO is unavailable.
    
    Supported element types:
    - button
    - input_field
    - menu
    - dropdown
    - checkbox
    - link
    - icon
    """
    
    # Mapping from COCO classes to UI elements (for general object detection)
    # In production, you'd use a UI-specific trained model
    UI_CLASS_MAPPING = {
        # These are approximations using general object detection
        # A properly trained UI detection model would have actual UI classes
        'cell phone': 'button',  # Rectangular objects
        'remote': 'button',
        'keyboard': 'input_field',
        'tv': 'menu',
        'laptop': 'container',
        'book': 'card',
        'clock': 'icon',
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        use_heuristics: bool = True
    ):
        """
        Initialize UI detector.
        
        Args:
            model_path: Path to custom YOLO model (None for pretrained)
            confidence_threshold: Minimum confidence for detections
            use_heuristics: Whether to use heuristic detection as fallback
        """
        self.confidence_threshold = confidence_threshold
        self.use_heuristics = use_heuristics
        self.model = None
        self._yolo_available = False
        
        # Try to load YOLO model
        try:
            from ultralytics import YOLO
            
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # Use pretrained model
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
            
            self._yolo_available = True
            print("YOLOv8 model loaded successfully")
            
        except ImportError:
            print("YOLOv8 not available. Using heuristic detection.")
            self._yolo_available = False
        except Exception as e:
            print(f"Failed to load YOLO model: {e}. Using heuristic detection.")
            self._yolo_available = False
    
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0
    ) -> list[DetectedElement]:
        """
        Detect UI elements in a frame.
        
        Args:
            frame: Video frame (BGR)
            frame_number: Current frame number
            
        Returns:
            List of detected UI elements
        """
        elements = []
        
        # Try YOLO detection first
        if self._yolo_available and self.model is not None:
            yolo_elements = self._detect_with_yolo(frame, frame_number)
            elements.extend(yolo_elements)
        
        # Add heuristic detections
        if self.use_heuristics:
            heuristic_elements = self._detect_with_heuristics(frame, frame_number)
            
            # Merge heuristic detections that don't overlap with YOLO
            for h_elem in heuristic_elements:
                if not self._overlaps_existing(h_elem, elements):
                    elements.append(h_elem)
        
        return elements
    
    def _detect_with_yolo(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> list[DetectedElement]:
        """
        Detect elements using YOLOv8.
        
        Args:
            frame: Video frame
            frame_number: Frame number
            
        Returns:
            List of detected elements
        """
        elements = []
        
        if self.model is None:
            return elements
        
        # Run inference
        results = self.model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                
                if conf < self.confidence_threshold:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # Get class name
                cls_id = int(boxes.cls[i])
                cls_name = self.model.names.get(cls_id, 'unknown')
                
                # Map to UI element type
                element_type = self.UI_CLASS_MAPPING.get(cls_name, 'generic')
                
                elements.append(DetectedElement(
                    element_type=element_type,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=conf,
                    frame_number=frame_number
                ))
        
        return elements
    
    def _detect_with_heuristics(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> list[DetectedElement]:
        """
        Detect UI elements using computer vision heuristics.
        
        Looks for:
        - Rectangular shapes (buttons, inputs)
        - High contrast regions
        - Text-like areas
        
        Args:
            frame: Video frame
            frame_number: Frame number
            
        Returns:
            List of detected elements
        """
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        height, width = frame.shape[:2]
        min_area = (width * height) * 0.001  # Min 0.1% of frame
        max_area = (width * height) * 0.3    # Max 30% of frame
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (min_area < area < max_area):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (UI elements are usually wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            
            if not (0.2 < aspect_ratio < 10):
                continue
            
            # Classify element type based on shape characteristics
            element_type = self._classify_element(frame, x, y, w, h)
            
            # Calculate confidence based on how "rectangular" the contour is
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            confidence = min(0.8, rectangularity + 0.3)
            
            elements.append(DetectedElement(
                element_type=element_type,
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
                confidence=confidence,
                frame_number=frame_number
            ))
        
        return elements
    
    def _classify_element(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> str:
        """
        Classify element type based on visual characteristics.
        
        Args:
            frame: Video frame
            x, y, w, h: Bounding box
            
        Returns:
            Element type string
        """
        aspect_ratio = w / h if h > 0 else 1
        
        # Extract region
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 'generic'
        
        # Analyze color variance
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray_roi)
        
        # Analyze edge density
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Classification rules
        if 2 < aspect_ratio < 8 and h < 60:
            if variance < 500:
                return 'button'
            else:
                return 'input_field'
        
        if 0.8 < aspect_ratio < 1.2 and w < 50:
            return 'icon'
        
        if aspect_ratio > 3 and variance > 1000:
            return 'menu'
        
        if h < 40 and edge_density < 0.1:
            return 'link'
        
        return 'generic'
    
    def _overlaps_existing(
        self,
        element: DetectedElement,
        existing: list[DetectedElement],
        iou_threshold: float = 0.5
    ) -> bool:
        """
        Check if element overlaps significantly with existing detections.
        
        Args:
            element: Element to check
            existing: List of existing elements
            iou_threshold: IoU threshold for overlap
            
        Returns:
            True if overlapping
        """
        for ex in existing:
            iou = self._calculate_iou(element, ex)
            if iou > iou_threshold:
                return True
        return False
    
    def _calculate_iou(
        self,
        elem1: DetectedElement,
        elem2: DetectedElement
    ) -> float:
        """
        Calculate Intersection over Union between two elements.
        
        Args:
            elem1, elem2: Elements to compare
            
        Returns:
            IoU value (0-1)
        """
        # Calculate intersection
        x1 = max(elem1.x1, elem2.x1)
        y1 = max(elem1.y1, elem2.y1)
        x2 = min(elem1.x2, elem2.x2)
        y2 = min(elem1.y2, elem2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = elem1.area
        area2 = elem2.area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_elements_at_position(
        self,
        elements: list[DetectedElement],
        x: int,
        y: int
    ) -> list[DetectedElement]:
        """
        Get all elements that contain a given position.
        
        Args:
            elements: List of detected elements
            x, y: Position to check
            
        Returns:
            List of elements containing the position
        """
        containing = []
        for elem in elements:
            if elem.x1 <= x <= elem.x2 and elem.y1 <= y <= elem.y2:
                containing.append(elem)
        return containing
    
    @property
    def is_yolo_available(self) -> bool:
        """Check if YOLO model is available."""
        return self._yolo_available

