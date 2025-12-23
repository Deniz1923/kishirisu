"""
Unit tests for object detection module.
"""

import numpy as np
import pytest

from stereo_vision import BoundingBox, Detection, DummyDetector


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_creation(self) -> None:
        box = BoundingBox(cx=100, cy=100, w=50, h=30)
        assert box.cx == 100
        assert box.cy == 100
        assert box.w == 50
        assert box.h == 30

    def test_corners(self) -> None:
        box = BoundingBox(cx=100, cy=100, w=50, h=30)
        x1, y1, x2, y2 = box.corners
        
        assert x1 == 75   # 100 - 50/2
        assert y1 == 85   # 100 - 30/2
        assert x2 == 125  # 100 + 50/2
        assert y2 == 115  # 100 + 30/2

    def test_area(self) -> None:
        box = BoundingBox(cx=0, cy=0, w=100, h=50)
        assert box.area == 5000

    def test_from_corners(self) -> None:
        box = BoundingBox.from_corners(50, 50, 150, 100)
        
        assert box.cx == 100
        assert box.cy == 75
        assert box.w == 100
        assert box.h == 50

    def test_iou_identical(self) -> None:
        box = BoundingBox(cx=100, cy=100, w=50, h=50)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        box1 = BoundingBox(cx=0, cy=0, w=10, h=10)
        box2 = BoundingBox(cx=100, cy=100, w=10, h=10)
        assert box1.iou(box2) == 0.0

    def test_iou_partial(self) -> None:
        box1 = BoundingBox(cx=100, cy=100, w=100, h=100)
        box2 = BoundingBox(cx=150, cy=100, w=100, h=100)
        
        iou = box1.iou(box2)
        assert 0 < iou < 1


class TestDetection:
    """Tests for Detection dataclass."""

    def test_creation(self) -> None:
        bbox = BoundingBox(cx=100, cy=100, w=50, h=50)
        det = Detection(label="ball", bbox=bbox, confidence=0.95)
        
        assert det.label == "ball"
        assert det.confidence == 0.95
        assert det.track_id is None

    def test_convenience_properties(self) -> None:
        bbox = BoundingBox(cx=100, cy=200, w=50, h=30)
        det = Detection(label="test", bbox=bbox, confidence=0.5)
        
        assert det.x == 100
        assert det.y == 200
        assert det.width == 50
        assert det.height == 30

    def test_draw_on(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(cx=320, cy=240, w=100, h=100)
        det = Detection(label="ball", bbox=bbox, confidence=0.9)
        
        result = det.draw_on(frame)
        
        # Should return the same frame object
        assert result is frame
        # Frame should be modified (not all zeros)
        assert frame.sum() > 0

    def test_with_track_id(self) -> None:
        bbox = BoundingBox(cx=100, cy=100, w=50, h=50)
        det = Detection(label="person", bbox=bbox, confidence=0.8, track_id=42)
        
        assert det.track_id == 42


class TestDummyDetector:
    """Tests for DummyDetector placeholder."""

    def test_creation(self) -> None:
        detector = DummyDetector(label="ball")
        assert detector is not None

    def test_detect_empty_frame(self) -> None:
        detector = DummyDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(frame)
        
        assert isinstance(detections, list)
        # Empty/black frame should have no detections
        assert len(detections) == 0

    def test_detect_colored_object(self) -> None:
        detector = DummyDetector(
            label="red_ball",
            hsv_lower=(0, 100, 100),
            hsv_upper=(20, 255, 255),
            min_area=100,
        )
        
        # Create frame with a red circle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.circle(frame, (320, 240), 50, (0, 0, 255), -1)  # Red circle
        
        detections = detector.detect(frame)
        
        assert len(detections) > 0
        assert detections[0].label == "red_ball"

    def test_detect_batch(self) -> None:
        detector = DummyDetector()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        
        results = detector.detect_batch(frames)
        
        assert len(results) == 3

    def test_set_color_range(self) -> None:
        detector = DummyDetector()
        detector.set_color_range((35, 100, 100), (85, 255, 255))  # Green range
        
        # Should not raise
        assert detector is not None

    def test_max_detections(self) -> None:
        detector = DummyDetector(max_detections=2, min_area=10)
        
        # Create frame with many red spots
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        for i in range(5):
            cv2.circle(frame, (100 + i * 100, 240), 20, (0, 0, 255), -1)
        
        detections = detector.detect(frame)
        
        assert len(detections) <= 2
