"""
Stereo Vision Module for Ball Catcher Robot
============================================

This package provides depth calculation from stereo cameras with support for:
- Mock camera using single webcam (for development/testing)
- Real stereo camera support (for deployment)
- Modular object detection interface (for YOLO integration)

Architecture:
    StereoCapture (abstract) → MockStereoCamera / RealStereoCamera
    DepthCalculator          → Computes depth from stereo pairs
    ObjectDetector (abstract) → YOLODetector (to be implemented by friend)

Usage:
    from stereo_vision import MockStereoCamera, DepthCalculator, StereoConfig
    
    config = StereoConfig()
    camera = MockStereoCamera(config)
    depth_calc = DepthCalculator(config)
    
    left, right = camera.capture()
    depth_map = depth_calc.compute(left, right)
    depth_at_point = depth_calc.get_depth_at(depth_map, x=320, y=240)
"""

from .config import StereoConfig, detect_camera_capabilities
from .capture import StereoCapture
from .mock_camera import MockStereoCamera
from .depth import DepthCalculator
from .detection import Detection, ObjectDetector, DummyDetector

__all__ = [
    "StereoConfig",
    "detect_camera_capabilities",
    "StereoCapture", 
    "MockStereoCamera",
    "DepthCalculator",
    "Detection",
    "ObjectDetector",
    "DummyDetector",
]

__version__ = "0.1.0"
