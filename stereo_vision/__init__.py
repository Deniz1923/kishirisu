"""
Kishirisu Stereo Vision
=======================

High-performance stereo depth estimation for robotics.

Quick Start::

    from stereo_vision import StereoConfig, MockStereoCamera, DepthCalculator

    config = StereoConfig.from_camera()
    with MockStereoCamera(config) as camera:
        left, right = camera.capture()
        result = DepthCalculator(config).compute(left, right)
        print(f"Center depth: {result.at_center():.0f}mm")

Modules:
    config: Camera configuration and SGBM parameters
    capture: Abstract capture interface
    mock_camera: Single-webcam mock stereo
    depth: SGBM depth calculation
    detection: Object detection interface
"""

__version__ = "1.0.0"

# Configuration
from .config import (
    Resolution,
    QualityPreset,
    SGBMParams,
    StereoConfig,
    detect_camera_capabilities,
)

# Capture
from .capture import (
    CameraPosition,
    StereoCapture,
    BaseStereoCamera,
)

# Mock Camera
from .mock_camera import (
    MockStereoCamera,
    ObjectShape,
    SceneObject,
    SceneInfo,
    NoiseMode,
)

# Depth
from .depth import (
    DepthZone,
    DepthStats,
    DepthResult,
    DepthCalculator,
)

# Detection
from .detection import (
    BoundingBox,
    Detection,
    Detector,
    ObjectDetector,
    BallDetector,
    DummyDetector,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "Resolution",
    "QualityPreset",
    "SGBMParams",
    "StereoConfig",
    "detect_camera_capabilities",
    # Capture
    "CameraPosition",
    "StereoCapture",
    "BaseStereoCamera",
    # Mock Camera
    "MockStereoCamera",
    "ObjectShape",
    "SceneObject",
    "SceneInfo",
    "NoiseMode",
    # Depth
    "DepthZone",
    "DepthStats",
    "DepthResult",
    "DepthCalculator",
    # Detection
    "BoundingBox",
    "Detection",
    "Detector",
    "ObjectDetector",
    "BallDetector",
    "DummyDetector",
]
