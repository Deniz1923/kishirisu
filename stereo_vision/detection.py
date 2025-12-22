"""
Object Detection Interface
===========================

This module defines the interface for object detection that your friend
will implement with YOLO.

The design philosophy is:
1. Define a clear, minimal interface (ObjectDetector abstract class)
2. Define a standard Detection dataclass for results
3. Provide a DummyDetector for testing until YOLO is ready

Your friend should subclass ObjectDetector and implement the detect() method
using their YOLO model. The rest of the pipeline (stereo capture, depth
calculation) will work unchanged.

Example YOLO implementation (for your friend):
---------------------------------------------
    from stereo_vision import Detection, ObjectDetector
    from ultralytics import YOLO
    
    class YOLODetector(ObjectDetector):
        def __init__(self, model_path: str):
            self.model = YOLO(model_path)
        
        def detect(self, frame: np.ndarray) -> list[Detection]:
            results = self.model(frame)
            detections = []
            for r in results[0].boxes:
                x, y, w, h = r.xywh[0].cpu().numpy()
                detections.append(Detection(
                    label=self.model.names[int(r.cls)],
                    x=int(x), y=int(y),
                    width=int(w), height=int(h),
                    confidence=float(r.conf)
                ))
            return detections
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class Detection:
    """
    Represents a detected object in an image.
    
    This is the standard output format that all detectors must use.
    This allows the depth calculation pipeline to work with any
    detection backend (YOLO, SSD, manual annotation, etc.)
    
    Attributes:
        label: Class name of detected object (e.g., "ball", "person")
        x: Center X coordinate of bounding box in pixels
        y: Center Y coordinate of bounding box in pixels
        width: Width of bounding box in pixels
        height: Height of bounding box in pixels
        confidence: Detection confidence score (0.0 to 1.0)
        
    Properties:
        bbox: Returns (x1, y1, x2, y2) corner coordinates
        area: Returns bounding box area in pixels
        
    Example:
        >>> det = Detection(label="ball", x=320, y=240, width=50, height=50, confidence=0.95)
        >>> print(f"Found {det.label} at ({det.x}, {det.y}) with {det.confidence:.0%} confidence")
        Found ball at (320, 240) with 95% confidence
    """
    label: str
    x: int           # Center X
    y: int           # Center Y
    width: int       # Bounding box width
    height: int      # Bounding box height
    confidence: float
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """
        Get bounding box as (x1, y1, x2, y2) corner coordinates.
        
        Returns:
            Tuple of (left, top, right, bottom) pixel coordinates
        """
        x1 = self.x - self.width // 2
        y1 = self.y - self.height // 2
        x2 = self.x + self.width // 2
        y2 = self.y + self.height // 2
        return (x1, y1, x2, y2)
    
    @property
    def area(self) -> int:
        """Get bounding box area in square pixels."""
        return self.width * self.height
    
    def draw_on(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_label: bool = True
    ) -> np.ndarray:
        """
        Draw this detection on a frame.
        
        Args:
            frame: Image to draw on (will be modified in place)
            color: BGR color for the bounding box
            thickness: Line thickness
            show_label: Whether to show label and confidence
            
        Returns:
            The frame with detection drawn (same as input)
        """
        x1, y1, x2, y2 = self.bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label and confidence
        if show_label:
            label_text = f"{self.label}: {self.confidence:.2f}"
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1  # Filled
            )
            
            # Draw text
            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                1
            )
        
        return frame


class ObjectDetector(ABC):
    """
    Abstract base class for object detection.
    
    YOUR FRIEND should subclass this and implement detect() with YOLO.
    
    This abstraction allows the stereo vision pipeline to work with
    any detection backend: YOLO, SSD, Faster R-CNN, or even manual
    annotation for testing.
    
    The interface is intentionally minimal to allow flexibility in
    implementation. The only required method is detect().
    
    Example Implementation:
        >>> class YOLODetector(ObjectDetector):
        ...     def __init__(self, model_path: str):
        ...         self.model = load_yolo_model(model_path)
        ...     
        ...     def detect(self, frame: np.ndarray) -> List[Detection]:
        ...         results = self.model.predict(frame)
        ...         return [Detection(...) for r in results]
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.
        
        This method must be implemented by all subclasses.
        
        Args:
            frame: Input image as numpy array (BGR format, HxWx3)
                  This is the standard OpenCV image format.
        
        Returns:
            List of Detection objects found in the frame.
            Return empty list if no objects detected.
            
        Note:
            - Implementations should be as fast as possible for real-time use
            - Consider running on GPU if available
            - Filter by confidence threshold before returning
        """
        pass
    
    def detect_in_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> List[Detection]:
        """
        Detect objects only within a Region of Interest.
        
        This is a convenience method that crops the frame to the ROI,
        runs detection, and adjusts coordinates back to full frame.
        
        Override this method if your detector has native ROI support
        for better performance.
        
        Args:
            frame: Full input image
            roi: (x1, y1, x2, y2) region of interest
            
        Returns:
            List of Detections with coordinates in full frame space
        """
        x1, y1, x2, y2 = roi
        
        # Crop to ROI
        roi_frame = frame[y1:y2, x1:x2]
        
        # Detect in cropped region
        detections = self.detect(roi_frame)
        
        # Adjust coordinates to full frame
        adjusted = []
        for det in detections:
            adjusted.append(Detection(
                label=det.label,
                x=det.x + x1,
                y=det.y + y1,
                width=det.width,
                height=det.height,
                confidence=det.confidence
            ))
        
        return adjusted


class DummyDetector(ObjectDetector):
    """
    Simple color-based detector for testing.
    
    This is a PLACEHOLDER detector that finds colored objects using
    HSV color filtering and contour detection. It's NOT meant for
    production use - it's just for testing the pipeline until your
    friend implements the YOLO detector.
    
    Default configuration detects orange/red objects (like typical balls).
    Modify the HSV range for different colors.
    
    Example:
        >>> # Create detector for orange balls
        >>> detector = DummyDetector(label="ball")
        >>> 
        >>> # Get detections
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     print(f"Found {det.label} at ({det.x}, {det.y})")
    """
    
    def __init__(
        self,
        label: str = "ball",
        hsv_lower: Tuple[int, int, int] = (0, 100, 100),
        hsv_upper: Tuple[int, int, int] = (20, 255, 255),
        min_area: int = 500,
        max_detections: int = 5
    ):
        """
        Initialize the dummy detector.
        
        Args:
            label: Label to assign to detected objects
            hsv_lower: Lower bound of HSV color range (H: 0-179, S: 0-255, V: 0-255)
            hsv_upper: Upper bound of HSV color range
            min_area: Minimum contour area to be considered a detection
            max_detections: Maximum number of detections to return
            
        Common HSV ranges:
            - Orange: lower=(0, 100, 100), upper=(20, 255, 255)
            - Red: lower=(0, 100, 100), upper=(10, 255, 255) OR (170, 100, 100)-(180, 255, 255)
            - Green: lower=(40, 100, 100), upper=(80, 255, 255)
            - Blue: lower=(100, 100, 100), upper=(140, 255, 255)
            - Yellow: lower=(20, 100, 100), upper=(40, 255, 255)
        """
        self._label = label
        self._hsv_lower = np.array(hsv_lower)
        self._hsv_upper = np.array(hsv_upper)
        self._min_area = min_area
        self._max_detections = max_detections
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect colored objects in the frame.
        
        The algorithm:
        1. Convert frame to HSV color space
        2. Create mask for target color range
        3. Find contours in the mask
        4. Filter by minimum area
        5. Return bounding boxes as Detections
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of Detection objects for found colored regions
        """
        # Step 1: Convert to HSV color space
        # HSV (Hue, Saturation, Value) is better for color detection
        # than RGB because it separates color from brightness
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Step 2: Create binary mask for the target color
        # Pixels within the HSV range become white (255), others black (0)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        
        # Step 3: Clean up the mask with morphological operations
        # This removes noise (small white/black dots)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        # Step 4: Find contours (outlines of white regions)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,       # Only outer contours
            cv2.CHAIN_APPROX_SIMPLE  # Compress to key points
        )
        
        # Step 5: Convert contours to Detections
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip small contours (noise)
            if area < self._min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Create Detection
            # Confidence based on how circular the contour is
            # (balls should be roughly circular)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            confidence = min(1.0, circularity)  # Clamp to max 1.0
            
            detections.append(Detection(
                label=self._label,
                x=center_x,
                y=center_y,
                width=w,
                height=h,
                confidence=confidence
            ))
        
        # Sort by area (largest first) and limit count
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections[:self._max_detections]
    
    def set_color_range(
        self,
        hsv_lower: Tuple[int, int, int],
        hsv_upper: Tuple[int, int, int]
    ) -> None:
        """
        Update the HSV color range to detect.
        
        Args:
            hsv_lower: Lower bound (H, S, V)
            hsv_upper: Upper bound (H, S, V)
        """
        self._hsv_lower = np.array(hsv_lower)
        self._hsv_upper = np.array(hsv_upper)
