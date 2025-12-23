"""
Object Detection Interface
===========================

Type-safe detection interface designed for YOLO integration.

This module provides:
- BoundingBox: Immutable bounding box with geometry utilities
- Detection: Immutable detection result
- Detector: Protocol for detection backends
- DummyDetector: Color-based placeholder for testing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Immutable bounding box with center + size representation.

    Attributes:
        cx: Center X coordinate
        cy: Center Y coordinate
        w: Width
        h: Height
    """

    cx: int
    cy: int
    w: int
    h: int

    @property
    def corners(self) -> tuple[int, int, int, int]:
        """Get (x1, y1, x2, y2) corner coordinates."""
        return (
            self.cx - self.w // 2,
            self.cy - self.h // 2,
            self.cx + self.w // 2,
            self.cy + self.h // 2,
        )

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        return self.w * self.h

    @classmethod
    def from_corners(cls, x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
        """Create from corner coordinates."""
        return cls(
            cx=(x1 + x2) // 2,
            cy=(y1 + y2) // 2,
            w=x2 - x1,
            h=y2 - y1,
        )

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection over Union with another box."""
        x1a, y1a, x2a, y2a = self.corners
        x1b, y1b, x2b, y2b = other.corners

        xi1 = max(x1a, x1b)
        yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b)
        yi2 = min(y2a, y2b)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    """
    Immutable detection result.

    Attributes:
        label: Class name (e.g., "ball", "person")
        bbox: Bounding box
        confidence: Detection confidence (0.0 to 1.0)
        track_id: Optional tracking ID for multi-frame tracking
    """

    label: str
    bbox: BoundingBox
    confidence: float
    track_id: int | None = None

    # Convenience properties for backward compatibility
    @property
    def x(self) -> int:
        """Center X coordinate."""
        return self.bbox.cx

    @property
    def y(self) -> int:
        """Center Y coordinate."""
        return self.bbox.cy

    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.bbox.w

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.bbox.h

    def draw_on(
        self,
        frame: npt.NDArray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_label: bool = True,
    ) -> npt.NDArray:
        """
        Draw detection on frame.

        Args:
            frame: Image to draw on (modified in place)
            color: BGR color for bounding box
            thickness: Line thickness
            show_label: Whether to show label text

        Returns:
            The frame (same object as input)
        """
        x1, y1, x2, y2 = self.bbox.corners

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if show_label:
            text = f"{self.label}: {self.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX

            (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)

            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 5), font, 0.5, (0, 0, 0), 1)

        return frame


class Detector(Protocol):
    """
    Protocol for object detection backends.

    Implement this with YOLO, SSD, or any other detector.
    """

    def detect(self, frame: npt.NDArray) -> list[Detection]:
        """
        Detect objects in a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            List of Detection objects
        """
        ...


# Backward compatibility alias
ObjectDetector = Detector


class DummyDetector:
    """
    Simple color-based detector for testing.

    NOT for production - just a placeholder until YOLO is integrated.
    Detects colored objects using HSV filtering and contour analysis.

    Example:
        >>> detector = DummyDetector(label="ball")
        >>> detections = detector.detect(frame)
    """

    __slots__ = ("_label", "_hsv_lower", "_hsv_upper", "_min_area", "_max_count")

    def __init__(
        self,
        label: str = "ball",
        hsv_lower: tuple[int, int, int] = (0, 100, 100),
        hsv_upper: tuple[int, int, int] = (20, 255, 255),
        min_area: int = 500,
        max_detections: int = 5,
    ) -> None:
        """
        Initialize color detector.

        Args:
            label: Label for detected objects
            hsv_lower: Lower HSV bound (H: 0-179, S/V: 0-255)
            hsv_upper: Upper HSV bound
            min_area: Minimum contour area
            max_detections: Maximum detections to return
        """
        self._label = label
        self._hsv_lower = np.array(hsv_lower)
        self._hsv_upper = np.array(hsv_upper)
        self._min_area = min_area
        self._max_count = max_detections

    def detect(self, frame: npt.NDArray) -> list[Detection]:
        """Detect colored objects in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Circularity as confidence proxy
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

            detections.append(
                Detection(
                    label=self._label,
                    bbox=BoundingBox(cx=x + w // 2, cy=y + h // 2, w=w, h=h),
                    confidence=min(1.0, circularity),
                )
            )

        # Sort by area, limit count
        detections.sort(key=lambda d: d.bbox.area, reverse=True)
        return detections[: self._max_count]

    def detect_batch(self, frames: Sequence[npt.NDArray]) -> list[list[Detection]]:
        """Detect in multiple frames."""
        return [self.detect(f) for f in frames]

    def set_color_range(
        self,
        hsv_lower: tuple[int, int, int],
        hsv_upper: tuple[int, int, int],
    ) -> None:
        """Update HSV color range."""
        self._hsv_lower = np.array(hsv_lower)
        self._hsv_upper = np.array(hsv_upper)
