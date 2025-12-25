"""
Kishirisu Stereo Vision
=======================

High-performance stereo depth estimation for robotics.

Quick Start (Mock Camera for Testing)::

    from stereo_vision import StereoConfig, MockStereoCamera, DepthCalculator

    config = StereoConfig.from_camera()
    with MockStereoCamera(config) as camera:
        left, right = camera.capture()
        result = DepthCalculator(config).compute(left, right)
        print(f"Center depth: {result.at_center():.0f}mm")

Real Camera Usage::

    from stereo_vision import StereoConfig, RealStereoCamera, DepthCalculator

    config = StereoConfig.from_camera(camera_index=0)
    with RealStereoCamera(config) as camera:
        left, right = camera.capture()
        result = DepthCalculator(config).compute(left, right)
        print(f"Center depth: {result.at_center():.0f}mm")

Modules:
    config: Camera configuration and SGBM parameters
    capture: Abstract capture interface
    mock_camera: Synthetic stereo data for testing
    real_camera: Real hardware stereo capture
    depth: SGBM depth calculation
    detection: Object detection interface
"""

__version__ = "1.0.0"

# Configuration
from .config import (
    Resolution,
    QualityPreset,
    SGBMParams,
    DepthFilterParams,
    StereoConfig,
    detect_camera_capabilities,
)

# Capture
from .capture import (
    CameraPosition,
    StereoCapture,
    BaseStereoCamera,
)

# Mock Camera (for testing)
from .mock_camera import (
    MockStereoCamera,
    ObjectShape,
    SceneObject,
    SceneInfo,
    NoiseMode,
)

# Real Camera (for production)
from .real_camera import (
    RealStereoCamera,
    CaptureMode,
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

# Tracking
from .tracking import (
    Position3D,
    Velocity3D,
    TrackState,
    BallTracker,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "Resolution",
    "QualityPreset",
    "SGBMParams",
    "DepthFilterParams",
    "StereoConfig",
    "detect_camera_capabilities",
    # Capture
    "CameraPosition",
    "StereoCapture",
    "BaseStereoCamera",
    # Mock Camera (testing)
    "MockStereoCamera",
    "ObjectShape",
    "SceneObject",
    "SceneInfo",
    "NoiseMode",
    # Real Camera (production)
    "RealStereoCamera",
    "CaptureMode",
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
    # Tracking
    "Position3D",
    "Velocity3D",
    "TrackState",
    "BallTracker",
]
