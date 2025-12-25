"""
Stereo Camera Configuration
============================

Immutable configuration dataclasses for stereo camera systems.
All measurements are in millimeters (mm) unless otherwise specified.

This module provides:
- Resolution: Type-safe resolution representation
- QualityPreset: Predefined quality/speed tradeoffs
- SGBMParams: Stereo matching algorithm parameters
- StereoConfig: Main configuration with validation and factory methods
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from enum import Enum, auto

import cv2
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
from typing import NamedTuple, Self




class Resolution(NamedTuple):
    """Type-safe resolution representation."""

    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio."""
        return self.width / self.height

    def scaled(self, factor: float) -> Resolution:
        """Return a new Resolution scaled by factor."""
        return Resolution(int(self.width * factor), int(self.height * factor))

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    @classmethod
    def parse(cls, s: str) -> Resolution:
        """Parse 'WxH' string to Resolution."""
        w, h = s.lower().split("x")
        return cls(int(w), int(h))


class QualityPreset(Enum):
    """Predefined quality/performance tradeoffs."""

    FAST = auto()  # Low latency, lower accuracy
    BALANCED = auto()  # Good tradeoff for most use cases
    QUALITY = auto()  # Best accuracy, higher latency


class SGBMParams(NamedTuple):
    """Semi-Global Block Matching algorithm parameters.

    These control the quality and speed of stereo depth computation.
    Use presets via SGBMParams.for_preset() for common configurations.
    """

    min_disparity: int = 0
    num_disparities: int = 64  # Must be divisible by 16
    block_size: int = 11  # Must be odd, >= 5
    p1: int | None = None  # Auto-calculate if None
    p2: int | None = None  # Auto-calculate if None
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    @classmethod
    def for_preset(cls, preset: QualityPreset) -> SGBMParams:
        """Get SGBM parameters optimized for a quality preset."""
        match preset:
            case QualityPreset.FAST:
                return cls(
                    num_disparities=48,
                    block_size=5,
                    uniqueness_ratio=5,
                    speckle_window_size=50,
                    mode=cv2.STEREO_SGBM_MODE_SGBM,
                )
            case QualityPreset.QUALITY:
                return cls(
                    num_disparities=128,
                    block_size=11,
                    uniqueness_ratio=15,
                    speckle_window_size=200,
                    speckle_range=16,
                    mode=cv2.STEREO_SGBM_MODE_HH4,
                )
            case _:  # BALANCED
                return cls()  # Defaults are balanced

    def get_p1(self) -> int:
        """Get P1 penalty, auto-calculating if not set."""
        if self.p1 is not None:
            return self.p1
        return 8 * 3 * self.block_size**2

    def get_p2(self) -> int:
        """Get P2 penalty, auto-calculating if not set."""
        if self.p2 is not None:
            return self.p2
        return 32 * 3 * self.block_size**2


class DepthFilterParams(NamedTuple):
    """Depth filtering and refinement parameters.

    These control post-processing for improved depth accuracy.
    All filters are optional and can be combined.

    Attributes:
        use_wls_filter: Enable WLS edge-preserving filter (recommended)
        wls_lambda: WLS regularization strength (higher = smoother)
        wls_sigma: WLS color sensitivity (higher = more edge-preserving)
        use_bilateral: Enable bilateral pre-filtering of input images
        bilateral_d: Bilateral filter kernel diameter
        bilateral_sigma_color: Bilateral color sigma
        bilateral_sigma_space: Bilateral spatial sigma
        left_right_check: Enable left-right disparity consistency check
        lr_threshold: Max disparity difference for LR check (pixels)
        temporal_alpha: Temporal smoothing factor (0=off, 0.3=moderate, 0.7=strong)
    """

    # WLS (Weighted Least Squares) filter - reduces noise while preserving edges
    use_wls_filter: bool = True
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.5

    # Bilateral pre-filtering - smooths input while preserving edges
    use_bilateral: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Left-right consistency check - eliminates half-occluded regions
    left_right_check: bool = True
    lr_threshold: int = 1

    # Temporal smoothing - reduces frame-to-frame jitter
    temporal_alpha: float = 0.0  # 0 = disabled

    @classmethod
    def for_preset(cls, preset: QualityPreset) -> "DepthFilterParams":
        """Get filter parameters optimized for a quality preset."""
        match preset:
            case QualityPreset.FAST:
                return cls(
                    use_wls_filter=False,
                    use_bilateral=False,
                    left_right_check=False,
                    temporal_alpha=0.0,
                )
            case QualityPreset.QUALITY:
                return cls(
                    use_wls_filter=True,
                    wls_lambda=16000.0,
                    wls_sigma=2.0,
                    use_bilateral=True,
                    left_right_check=True,
                    lr_threshold=1,
                    temporal_alpha=0.2,
                )
            case _:  # BALANCED
                return cls()  # Defaults are balanced

def detect_camera_capabilities(camera_index: int = 0) -> dict:
    """
    Detect camera resolution and FPS capabilities.

    Opens the camera briefly to query its actual capabilities.
    Useful for auto-configuring the stereo system.

    Args:
        camera_index: OpenCV camera index (0 = default webcam)

    Returns:
        Dictionary with detected capabilities:
        - resolution: Resolution namedtuple
        - fps: Frames per second
        - backend: Camera backend name

    Raises:
        RuntimeError: If camera cannot be opened
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera {camera_index}. "
            "Ensure webcam is connected and not in use."
        )

    try:
        # Try common resolutions, highest first
        for width, height in [(1920, 1080), (1280, 720), (640, 480)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_w >= width:
                break

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        try:
            backend = cap.getBackendName()
        except Exception:
            backend = "Unknown"

        return {
            "resolution": Resolution(actual_w, actual_h),
            "fps": fps,
            "backend": backend,
            "camera_index": camera_index,
        }

    finally:
        cap.release()


@dataclass(frozen=True, slots=True)
class StereoConfig:
    """
    Immutable configuration for stereo vision system.

    Use factory methods for convenient construction:
    - StereoConfig.from_camera() - Auto-detect from hardware
    - StereoConfig.for_preset() - Use quality preset

    Attributes:
        resolution: Image dimensions (width, height)
        fps: Target frames per second
        focal_length_px: Camera focal length in pixels (auto-estimated if None)
        principal_point: Optical center (cx, cy), defaults to image center
        baseline_mm: Distance between stereo cameras in millimeters
        min_depth_mm: Minimum reliable depth measurement
        max_depth_mm: Maximum reliable depth measurement
        sgbm: SGBM algorithm parameters
        depth_filter: Depth filtering and refinement parameters

    Example:
        >>> config = StereoConfig.from_camera(baseline_mm=65.0)
        >>> print(f"Using {config.resolution} @ {config.fps}fps")
    """

    resolution: Resolution = field(default_factory=lambda: Resolution(640, 480))
    fps: float = 30.0
    focal_length_px: float | None = None
    principal_point: tuple[float, float] | None = None
    baseline_mm: float = 60.0
    min_depth_mm: float = 200.0
    max_depth_mm: float = 5000.0
    sgbm: SGBMParams = field(default_factory=SGBMParams)
    depth_filter: DepthFilterParams = field(default_factory=DepthFilterParams)


    # Private computed values (calculated in __post_init__)
    _focal: float = field(init=False, repr=False, compare=False)
    _cx: float = field(init=False, repr=False, compare=False)
    _cy: float = field(init=False, repr=False, compare=False)
    _depth_factor: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration and compute derived values."""
        # Validate resolution
        if self.resolution.width <= 0 or self.resolution.height <= 0:
            raise ValueError(f"Invalid resolution: {self.resolution}")

        # Validate baseline
        if self.baseline_mm <= 0:
            raise ValueError(f"Baseline must be positive: {self.baseline_mm}")

        # Validate SGBM params
        if self.sgbm.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16")
        if self.sgbm.block_size < 5 or self.sgbm.block_size % 2 == 0:
            raise ValueError("block_size must be odd and >= 5")

        # Compute focal length (estimate ~70° HFOV if not provided)
        focal = (
            self.focal_length_px
            if self.focal_length_px is not None
            else self.resolution.width * 0.82
        )
        object.__setattr__(self, "_focal", focal)

        # Compute principal point
        if self.principal_point is not None:
            cx, cy = self.principal_point
        else:
            cx = self.resolution.width / 2.0
            cy = self.resolution.height / 2.0
        object.__setattr__(self, "_cx", cx)
        object.__setattr__(self, "_cy", cy)

        # Pre-compute depth factor
        object.__setattr__(self, "_depth_factor", focal * self.baseline_mm)

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.resolution.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.resolution.height

    @property
    def focal(self) -> float:
        """Focal length in pixels (computed or explicit)."""
        return self._focal

    @property
    def cx(self) -> float:
        """Principal point X coordinate."""
        return self._cx

    @property
    def cy(self) -> float:
        """Principal point Y coordinate."""
        return self._cy

    @property
    def depth_factor(self) -> float:
        """Pre-computed focal_length * baseline for depth calculation."""
        return self._depth_factor

    def get_camera_matrix(self) -> "npt.NDArray[np.float64]":
        """Get 3x3 camera intrinsic matrix K."""
        return np.array(
            [
                [self._focal, 0, self._cx],
                [0, self._focal, self._cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_camera(
        cls,
        camera_index: int = 0,
        baseline_mm: float = 60.0,
        prefer_resolution: Resolution | None = None,
        preset: QualityPreset = QualityPreset.BALANCED,
    ) -> Self:
        """
        Create configuration by auto-detecting camera capabilities.

        Args:
            camera_index: OpenCV camera index
            baseline_mm: Stereo baseline (must be measured manually)
            prefer_resolution: Preferred resolution (uses detected if larger)
            preset: Quality preset for SGBM parameters

        Returns:
            StereoConfig with detected parameters
        """
        caps = detect_camera_capabilities(camera_index)
        resolution = caps["resolution"]

        if prefer_resolution is not None:
            if (
                prefer_resolution.width <= resolution.width
                and prefer_resolution.height <= resolution.height
            ):
                resolution = prefer_resolution

        return cls(
            resolution=resolution,
            fps=caps["fps"],
            baseline_mm=baseline_mm,
            sgbm=SGBMParams.for_preset(preset),
            depth_filter=DepthFilterParams.for_preset(preset),
        )

    @classmethod
    def for_preset(
        cls,
        preset: QualityPreset,
        resolution: Resolution = Resolution(640, 480),
        baseline_mm: float = 60.0,
    ) -> Self:
        """Create configuration with optimized parameters for a preset."""
        return cls(
            resolution=resolution,
            baseline_mm=baseline_mm,
            sgbm=SGBMParams.for_preset(preset),
            depth_filter=DepthFilterParams.for_preset(preset),
        )

    def scaled(self, factor: float) -> StereoConfig:
        """
        Create a new config scaled to a different resolution.

        Useful for processing at lower resolution for speed,
        then mapping results back to full resolution.
        """
        new_res = self.resolution.scaled(factor)
        return StereoConfig(
            resolution=new_res,
            fps=self.fps,
            focal_length_px=self._focal * factor,
            principal_point=(self._cx * factor, self._cy * factor),
            baseline_mm=self.baseline_mm,
            min_depth_mm=self.min_depth_mm,
            max_depth_mm=self.max_depth_mm,
            sgbm=self.sgbm,
            depth_filter=self.depth_filter,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary (for JSON storage)."""
        return {
            "resolution": [self.resolution.width, self.resolution.height],
            "fps": self.fps,
            "focal_length_px": self._focal,
            "principal_point": [self._cx, self._cy],
            "baseline_mm": self.baseline_mm,
            "min_depth_mm": self.min_depth_mm,
            "max_depth_mm": self.max_depth_mm,
            "sgbm": {
                "num_disparities": self.sgbm.num_disparities,
                "block_size": self.sgbm.block_size,
            },
            "depth_filter": {
                "use_wls_filter": self.depth_filter.use_wls_filter,
                "wls_lambda": self.depth_filter.wls_lambda,
                "wls_sigma": self.depth_filter.wls_sigma,
                "use_bilateral": self.depth_filter.use_bilateral,
                "left_right_check": self.depth_filter.left_right_check,
                "lr_threshold": self.depth_filter.lr_threshold,
                "temporal_alpha": self.depth_filter.temporal_alpha,
            },
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values (from to_dict())

        Returns:
            StereoConfig instance
        """
        resolution = Resolution(*data["resolution"])
        
        sgbm_data = data.get("sgbm", {})
        sgbm = SGBMParams(
            num_disparities=sgbm_data.get("num_disparities", 64),
            block_size=sgbm_data.get("block_size", 11),
        )
        
        pp = data.get("principal_point")
        principal_point = tuple(pp) if pp else None
        
        # Parse depth filter params
        df_data = data.get("depth_filter", {})
        depth_filter = DepthFilterParams(
            use_wls_filter=df_data.get("use_wls_filter", True),
            wls_lambda=df_data.get("wls_lambda", 8000.0),
            wls_sigma=df_data.get("wls_sigma", 1.5),
            use_bilateral=df_data.get("use_bilateral", False),
            left_right_check=df_data.get("left_right_check", True),
            lr_threshold=df_data.get("lr_threshold", 1),
            temporal_alpha=df_data.get("temporal_alpha", 0.0),
        )
        
        return cls(
            resolution=resolution,
            fps=data.get("fps", 30.0),
            focal_length_px=data.get("focal_length_px"),
            principal_point=principal_point,
            baseline_mm=data.get("baseline_mm", 60.0),
            min_depth_mm=data.get("min_depth_mm", 200.0),
            max_depth_mm=data.get("max_depth_mm", 5000.0),
            sgbm=sgbm,
            depth_filter=depth_filter,
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """
        Create configuration from JSON string.

        Args:
            json_str: JSON string (from to_json())

        Returns:
            StereoConfig instance
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_calibration(cls, path: str | Path) -> Self:
        """
        Load configuration from a YAML calibration file.
        
        Calibration file format:
            resolution: [1280, 720]
            baseline_mm: 65.0
            focal_length_px: 1050.0
            principal_point: [640.0, 360.0]
            min_depth_mm: 200.0
            max_depth_mm: 5000.0
        
        Args:
            path: Path to YAML calibration file
            
        Returns:
            StereoConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        from pathlib import Path
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Invalid calibration file format: {path}")
        
        return cls.from_dict(data)

    def save_calibration(self, path: str | Path) -> None:
        """
        Save configuration to a YAML calibration file.
        
        Args:
            path: Path to save YAML file (will create parent dirs)
        """
        from pathlib import Path
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

