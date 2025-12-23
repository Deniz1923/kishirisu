"""
Stereo Depth Calculator
========================

High-performance depth computation from stereo image pairs using
OpenCV's Semi-Global Block Matching (SGBM) algorithm.

Theory:
    depth = (focal_length × baseline) / disparity

Where disparity is the horizontal pixel difference between matching
points in left/right images.

This module provides:
- DepthResult: Immutable result container with depth map and statistics
- DepthCalculator: Main depth computation engine with multiple modes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import StereoConfig

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class DepthStats:
    """Statistics about a depth map."""

    min_mm: float
    max_mm: float
    median_mm: float
    mean_mm: float
    valid_ratio: float  # 0.0 to 1.0

    @classmethod
    def from_depth_map(cls, depth_map: npt.NDArray) -> DepthStats:
        """Compute statistics from depth map."""
        valid = depth_map[depth_map > 0]

        if len(valid) == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0)

        return cls(
            min_mm=float(np.min(valid)),
            max_mm=float(np.max(valid)),
            median_mm=float(np.median(valid)),
            mean_mm=float(np.mean(valid)),
            valid_ratio=len(valid) / depth_map.size,
        )


@dataclass(frozen=True, slots=True)
class DepthResult:
    """
    Immutable container for depth computation results.

    Attributes:
        depth_map: 2D array of depth values in millimeters (0 = invalid)
        stats: Computed statistics about the depth map
        disparity_map: Raw disparity values (optional, for debugging)
    """

    depth_map: npt.NDArray
    stats: DepthStats
    disparity_map: npt.NDArray | None = None

    def at(self, x: int, y: int, window_size: int = 5) -> float:
        """
        Get depth at pixel location using median filtering.

        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
            window_size: Size of median filter window (odd number)

        Returns:
            Depth in mm, or 0.0 if invalid
        """
        h, w = self.depth_map.shape
        if not (0 <= x < w and 0 <= y < h):
            return 0.0

        half = window_size // 2
        y1, y2 = max(0, y - half), min(h, y + half + 1)
        x1, x2 = max(0, x - half), min(w, x + half + 1)

        window = self.depth_map[y1:y2, x1:x2]
        valid = window[window > 0]

        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def at_center(self, window_size: int = 15) -> float:
        """Get depth at image center."""
        h, w = self.depth_map.shape
        return self.at(w // 2, h // 2, window_size)


class DepthCalculator:
    """
    High-performance stereo depth estimation.

    Uses OpenCV's SGBM algorithm with configurable parameters for
    quality/speed tradeoffs.

    Example:
        >>> calc = DepthCalculator(config)
        >>> result = calc.compute(left, right)
        >>> print(f"Center depth: {result.at_center():.0f}mm")
    """

    __slots__ = ("_config", "_stereo", "_depth_factor")

    def __init__(self, config: StereoConfig) -> None:
        """
        Initialize depth calculator.

        Args:
            config: StereoConfig with camera and SGBM parameters
        """
        self._config = config
        self._depth_factor = config.depth_factor
        self._stereo = self._create_stereo_matcher()

    def _create_stereo_matcher(self) -> cv2.StereoSGBM:
        """Create SGBM stereo matcher from config."""
        sgbm = self._config.sgbm

        return cv2.StereoSGBM_create(
            minDisparity=sgbm.min_disparity,
            numDisparities=sgbm.num_disparities,
            blockSize=sgbm.block_size,
            P1=sgbm.get_p1(),
            P2=sgbm.get_p2(),
            disp12MaxDiff=sgbm.disp12_max_diff,
            preFilterCap=sgbm.pre_filter_cap,
            uniquenessRatio=sgbm.uniqueness_ratio,
            speckleWindowSize=sgbm.speckle_window_size,
            speckleRange=sgbm.speckle_range,
            mode=sgbm.mode,
        )

    def compute(
        self,
        left: npt.NDArray,
        right: npt.NDArray,
        include_disparity: bool = False,
    ) -> DepthResult:
        """
        Compute depth map from stereo pair.

        Args:
            left: Left camera BGR image
            right: Right camera BGR image
            include_disparity: Include raw disparity in result

        Returns:
            DepthResult with depth map and statistics
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Compute disparity (fixed-point, scaled by 16)
        disparity_fp = self._stereo.compute(left_gray, right_gray)
        disparity = disparity_fp.astype(np.float32) / 16.0

        # Convert to depth
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_map = self._depth_factor / disparity

        # Filter invalid values
        depth_map = self._filter_invalid(depth_map, disparity)

        return DepthResult(
            depth_map=depth_map,
            stats=DepthStats.from_depth_map(depth_map),
            disparity_map=disparity if include_disparity else None,
        )

    def compute_fast(
        self,
        left: npt.NDArray,
        right: npt.NDArray,
        scale: float = 0.5,
    ) -> DepthResult:
        """
        Fast depth computation with downscaling.

        Processes at lower resolution then upscales result.

        Args:
            left: Left camera BGR image
            right: Right camera BGR image
            scale: Downscale factor (0.5 = half resolution)

        Returns:
            DepthResult at original resolution
        """
        h, w = left.shape[:2]
        new_size = (int(w * scale), int(h * scale))

        # Downscale
        left_small = cv2.resize(left, new_size, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right, new_size, interpolation=cv2.INTER_AREA)

        # Compute at low res
        left_gray = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)

        disparity_fp = self._stereo.compute(left_gray, right_gray)
        disparity = (disparity_fp.astype(np.float32) / 16.0) / scale

        # Convert to depth
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_small = self._depth_factor / disparity

        # Upscale
        depth_map = cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_map = self._filter_invalid(depth_map)

        return DepthResult(
            depth_map=depth_map,
            stats=DepthStats.from_depth_map(depth_map),
        )

    def _filter_invalid(
        self,
        depth_map: npt.NDArray,
        disparity: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Set invalid depth values to 0."""
        invalid = (
            np.isinf(depth_map)
            | np.isnan(depth_map)
            | (depth_map < self._config.min_depth_mm)
            | (depth_map > self._config.max_depth_mm)
        )

        if disparity is not None:
            invalid |= disparity <= 0

        result = depth_map.copy()
        result[invalid] = 0
        return result

    def get_depth_at(
        self,
        depth_map: npt.NDArray,
        x: int,
        y: int,
        window_size: int = 5,
    ) -> float:
        """
        Get depth at pixel with median filtering.

        Args:
            depth_map: Depth map array
            x: Pixel X coordinate
            y: Pixel Y coordinate
            window_size: Median filter window size

        Returns:
            Depth in mm, or 0.0 if invalid
        """
        h, w = depth_map.shape
        if not (0 <= x < w and 0 <= y < h):
            return 0.0

        half = window_size // 2
        y1, y2 = max(0, y - half), min(h, y + half + 1)
        x1, x2 = max(0, x - half), min(w, x + half + 1)

        window = depth_map[y1:y2, x1:x2]
        valid = window[window > 0]

        return float(np.median(valid)) if len(valid) > 0 else 0.0

    def pixel_to_3d(
        self, x: int, y: int, depth_mm: float
    ) -> tuple[float, float, float]:
        """
        Convert pixel + depth to 3D camera coordinates.

        Coordinate system:
        - X: right (positive = right of camera)
        - Y: down (positive = below camera center)
        - Z: forward (depth into scene)

        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
            depth_mm: Depth at pixel in mm

        Returns:
            (X, Y, Z) in millimeters
        """
        if depth_mm <= 0:
            return (0.0, 0.0, 0.0)

        x_3d = (x - self._config.cx) * depth_mm / self._config.focal
        y_3d = (y - self._config.cy) * depth_mm / self._config.focal
        z_3d = depth_mm

        return (x_3d, y_3d, z_3d)

    def visualize(
        self,
        depth_map: npt.NDArray,
        colormap: int = cv2.COLORMAP_JET,
    ) -> npt.NDArray:
        """
        Create colorized depth visualization.

        Closer objects = warm colors (red/orange)
        Farther objects = cool colors (blue/purple)
        Invalid = black

        Args:
            depth_map: Depth values in mm
            colormap: OpenCV colormap constant

        Returns:
            BGR visualization image
        """
        valid = depth_map > 0
        min_d, max_d = self._config.min_depth_mm, self._config.max_depth_mm

        # Normalize (inverted: close = high value = warm color)
        normalized = np.zeros(depth_map.shape, dtype=np.uint8)
        if np.any(valid):
            clipped = np.clip(depth_map, min_d, max_d)
            depth_norm = (max_d - clipped) / (max_d - min_d)
            normalized[valid] = (depth_norm[valid] * 255).astype(np.uint8)

        colored = cv2.applyColorMap(normalized, colormap)
        colored[~valid] = [0, 0, 0]

        return colored

    def get_depth_stats(self, depth_map: npt.NDArray) -> dict:
        """
        Get depth statistics as dictionary.

        Compatibility method - prefer using DepthStats directly.
        """
        stats = DepthStats.from_depth_map(depth_map)
        return {
            "min_depth": stats.min_mm,
            "max_depth": stats.max_mm,
            "median_depth": stats.median_mm,
            "mean_depth": stats.mean_mm,
            "valid_percent": stats.valid_ratio * 100,
        }

    @property
    def config(self) -> StereoConfig:
        """Get stereo configuration."""
        return self._config
