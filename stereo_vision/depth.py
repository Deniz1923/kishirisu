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


__all__ = [
    "DepthZone",
    "DepthStats",
    "DepthResult",
    "DepthCalculator",
]


@dataclass(frozen=True, slots=True)
class DepthZone:
    """Depth zone classification for verbose analysis."""

    name: str  # "near", "mid", "far"
    range_mm: tuple[float, float]
    pixel_count: int
    percentage: float

    def __str__(self) -> str:
        return f"{self.name.upper()}: {self.percentage:.1f}% ({self.pixel_count:,} px)"


@dataclass(frozen=True, slots=True)
class DepthStats:
    """Extended statistics about a depth map.
    
    Attributes:
        min_mm: Minimum valid depth
        max_mm: Maximum valid depth
        median_mm: Median depth value
        mean_mm: Mean depth value
        std_mm: Standard deviation of depth values
        p25_mm: 25th percentile (near boundary)
        p75_mm: 75th percentile (far boundary)
        valid_ratio: Fraction of valid pixels (0.0 to 1.0)
        confidence: Overall quality score (0.0 to 1.0)
        zones: Depth zone breakdown (near/mid/far)
    """

    min_mm: float
    max_mm: float
    median_mm: float
    mean_mm: float
    std_mm: float
    p25_mm: float
    p75_mm: float
    valid_ratio: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    zones: tuple[DepthZone, DepthZone, DepthZone] | None = None

    # Zone boundaries (class constants)
    NEAR_MAX_MM: float = 1000.0
    MID_MAX_MM: float = 3000.0

    @classmethod
    def from_depth_map(
        cls,
        depth_map: npt.NDArray,
        near_threshold_mm: float = 1000.0,
        far_threshold_mm: float = 3000.0,
    ) -> DepthStats:
        """Compute comprehensive statistics from depth map.
        
        Args:
            depth_map: 2D array of depth values in mm (0 = invalid)
            near_threshold_mm: Depth below this is "near" zone
            far_threshold_mm: Depth above this is "far" zone
        """
        valid = depth_map[depth_map > 0]
        total_pixels = depth_map.size

        if len(valid) == 0:
            empty_zone = DepthZone("empty", (0, 0), 0, 0.0)
            return cls(
                min_mm=0.0,
                max_mm=0.0,
                median_mm=0.0,
                mean_mm=0.0,
                std_mm=0.0,
                p25_mm=0.0,
                p75_mm=0.0,
                valid_ratio=0.0,
                confidence=0.0,
                zones=(empty_zone, empty_zone, empty_zone),
            )

        # Basic statistics
        min_d = float(np.min(valid))
        max_d = float(np.max(valid))
        median_d = float(np.median(valid))
        mean_d = float(np.mean(valid))
        std_d = float(np.std(valid))
        p25 = float(np.percentile(valid, 25))
        p75 = float(np.percentile(valid, 75))
        valid_ratio = len(valid) / total_pixels

        # Compute depth zones
        near_mask = valid < near_threshold_mm
        far_mask = valid >= far_threshold_mm
        mid_mask = ~near_mask & ~far_mask

        near_count = int(np.sum(near_mask))
        mid_count = int(np.sum(mid_mask))
        far_count = int(np.sum(far_mask))
        valid_count = len(valid)

        zones = (
            DepthZone(
                "near",
                (min_d, near_threshold_mm),
                near_count,
                100.0 * near_count / valid_count if valid_count > 0 else 0.0,
            ),
            DepthZone(
                "mid",
                (near_threshold_mm, far_threshold_mm),
                mid_count,
                100.0 * mid_count / valid_count if valid_count > 0 else 0.0,
            ),
            DepthZone(
                "far",
                (far_threshold_mm, max_d),
                far_count,
                100.0 * far_count / valid_count if valid_count > 0 else 0.0,
            ),
        )

        # Confidence score based on:
        # - Valid pixel ratio (50% weight)
        # - Low noise / std deviation (25% weight)
        # - Reasonable depth range (25% weight)
        valid_score = min(1.0, valid_ratio / 0.7)  # Normalize to 70% as "good"
        range_score = 1.0 if 100 < (max_d - min_d) < 4000 else 0.5
        noise_score = max(0.0, 1.0 - std_d / 1000.0)  # Lower std = higher score
        confidence = 0.5 * valid_score + 0.25 * range_score + 0.25 * noise_score

        return cls(
            min_mm=min_d,
            max_mm=max_d,
            median_mm=median_d,
            mean_mm=mean_d,
            std_mm=std_d,
            p25_mm=p25,
            p75_mm=p75,
            valid_ratio=valid_ratio,
            confidence=confidence,
            zones=zones,
        )

    def format_verbose(self) -> str:
        """Format stats as multi-line verbose string."""
        lines = [
            f"  Range: {self.min_mm:.0f} - {self.max_mm:.0f}mm",
            f"  Mean: {self.mean_mm:.0f}mm | Median: {self.median_mm:.0f}mm | Std: {self.std_mm:.0f}mm",
            f"  Percentiles: P25={self.p25_mm:.0f}mm | P75={self.p75_mm:.0f}mm",
            f"  Valid: {self.valid_ratio*100:.1f}% | Confidence: {self.confidence*100:.0f}%",
        ]
        if self.zones:
            lines.append(f"  Zones: {self.zones[0]} | {self.zones[1]} | {self.zones[2]}")
        return "\n".join(lines)



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
    High-performance stereo depth estimation with advanced filtering.

    Uses OpenCV's SGBM algorithm with configurable parameters for
    quality/speed tradeoffs. Supports optional post-processing:
    - WLS (Weighted Least Squares) edge-preserving filter
    - Left-right consistency checking
    - Bilateral pre-filtering
    - Temporal smoothing

    Example:
        >>> calc = DepthCalculator(config)
        >>> result = calc.compute(left, right)
        >>> print(f"Center depth: {result.at_center():.0f}mm")
    """

    __slots__ = (
        "_config",
        "_stereo_left",
        "_stereo_right",
        "_wls_filter",
        "_depth_factor",
        "_prev_depth",
    )

    def __init__(self, config: StereoConfig) -> None:
        """
        Initialize depth calculator.

        Args:
            config: StereoConfig with camera and SGBM parameters
        """
        self._config = config
        self._depth_factor = config.depth_factor
        self._stereo_left = self._create_stereo_matcher()
        self._stereo_right: cv2.StereoMatcher | None = None
        self._wls_filter = None
        self._prev_depth: npt.NDArray | None = None

        # Create right matcher and WLS filter if needed
        if config.depth_filter.use_wls_filter or config.depth_filter.left_right_check:
            self._setup_wls_filter()

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

    def _setup_wls_filter(self) -> None:
        """Create right matcher and WLS filter for disparity refinement."""
        try:
            # cv2.ximgproc is in opencv-contrib-python
            self._stereo_right = cv2.ximgproc.createRightMatcher(self._stereo_left)
            df = self._config.depth_filter
            self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._stereo_left)
            self._wls_filter.setLambda(df.wls_lambda)
            self._wls_filter.setSigmaColor(df.wls_sigma)
            self._wls_filter.setLRCthresh(df.lr_threshold)
        except AttributeError:
            # opencv-contrib-python not installed, disable WLS
            self._stereo_right = None
            self._wls_filter = None

    def compute(
        self,
        left: npt.NDArray,
        right: npt.NDArray,
        include_disparity: bool = False,
    ) -> DepthResult:
        """
        Compute depth map from stereo pair with optional filtering.

        Applies configured filters for improved accuracy:
        - Bilateral pre-filtering (if enabled)
        - WLS edge-preserving filter (if enabled)
        - Temporal smoothing (if enabled)

        Args:
            left: Left camera BGR image
            right: Right camera BGR image
            include_disparity: Include raw disparity in result

        Returns:
            DepthResult with depth map and statistics
        """
        df = self._config.depth_filter

        # Optional bilateral pre-filtering
        if df.use_bilateral:
            left = cv2.bilateralFilter(
                left, df.bilateral_d, df.bilateral_sigma_color, df.bilateral_sigma_space
            )
            right = cv2.bilateralFilter(
                right, df.bilateral_d, df.bilateral_sigma_color, df.bilateral_sigma_space
            )

        # Convert to grayscale
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Compute left-to-right disparity (fixed-point, scaled by 16)
        disparity_left_fp = self._stereo_left.compute(left_gray, right_gray)

        # Apply WLS filtering if available
        if self._wls_filter is not None and self._stereo_right is not None:
            # Compute right-to-left disparity for WLS
            disparity_right_fp = self._stereo_right.compute(right_gray, left_gray)

            # WLS filter combines L-R disparity for edge-preserving smoothing
            disparity_fp = self._wls_filter.filter(
                disparity_left_fp, left, disparity_right_fp
            )
        else:
            disparity_fp = disparity_left_fp

        disparity = disparity_fp.astype(np.float32) / 16.0

        # Convert to depth
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_map = self._depth_factor / disparity

        # Filter invalid values
        depth_map = self._filter_invalid(depth_map, disparity)

        # Apply temporal smoothing
        if df.temporal_alpha > 0 and self._prev_depth is not None:
            depth_map = self._apply_temporal_smoothing(depth_map, df.temporal_alpha)

        # Store for next frame's temporal smoothing
        self._prev_depth = depth_map.copy()

        return DepthResult(
            depth_map=depth_map,
            stats=DepthStats.from_depth_map(depth_map),
            disparity_map=disparity if include_disparity else None,
        )

    def _apply_temporal_smoothing(
        self, depth_map: npt.NDArray, alpha: float
    ) -> npt.NDArray:
        """Apply exponential moving average for temporal smoothing."""
        if self._prev_depth is None or self._prev_depth.shape != depth_map.shape:
            return depth_map

        # Only smooth where both frames have valid depth
        valid_both = (depth_map > 0) & (self._prev_depth > 0)
        result = depth_map.copy()

        # EMA: new = alpha * current + (1-alpha) * previous
        result[valid_both] = (
            alpha * depth_map[valid_both] + (1 - alpha) * self._prev_depth[valid_both]
        )

        return result

    def reset_temporal(self) -> None:
        """Reset temporal smoothing state (call when scene changes)."""
        self._prev_depth = None

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

        disparity_fp = self._stereo_left.compute(left_gray, right_gray)
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
