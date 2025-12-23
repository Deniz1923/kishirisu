"""
Mock Stereo Camera
===================

Generates synthetic stereo pairs with real depth variation for testing.

Architecture:
- SceneGenerator: Creates procedural depth maps and scenes
- StereoSynthesizer: Converts depth + left image to stereo pair
- MockStereoCamera: Main class combining the above
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .capture import BaseStereoCamera
from .config import Resolution, StereoConfig

if TYPE_CHECKING:
    import numpy.typing as npt


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True, slots=True)
class SceneObject:
    """A circular object in the scene at a specific depth."""

    x: int
    y: int
    radius: int
    depth_mm: float


@dataclass(slots=True)
class Scene:
    """A complete scene with depth map and image."""

    depth_map: npt.NDArray
    image: npt.NDArray
    objects: list[SceneObject] = field(default_factory=list)


# ============================================================================
# Scene Generation
# ============================================================================


class SceneGenerator(ABC):
    """Abstract base for scene generators."""

    @abstractmethod
    def generate(self, width: int, height: int, frame: int) -> Scene:
        """Generate a scene with depth map and image."""
        ...


class ProceduralSceneGenerator(SceneGenerator):
    """
    Generates procedural scenes with varying depth.

    Creates:
    - Background depth gradient (far at top, near at bottom)
    - Animated circular objects at different depths
    - Central focus object
    """

    def __init__(
        self,
        base_depth_mm: float = 1000.0,
        min_depth_mm: float = 300.0,
        max_depth_mm: float = 5000.0,
        num_objects: int = 6,
        seed: int = 42,
    ) -> None:
        self.base_depth = base_depth_mm
        self.min_depth = min_depth_mm
        self.max_depth = max_depth_mm
        self.num_objects = num_objects
        self._rng = np.random.default_rng(seed)

    def generate(self, width: int, height: int, frame: int) -> Scene:
        """Generate a complete scene."""
        depth_map, objects = self._generate_depth_map(width, height, frame)
        image = self._generate_image(width, height, depth_map, frame)
        return Scene(depth_map=depth_map, image=image, objects=objects)

    def _generate_depth_map(
        self, w: int, h: int, frame: int
    ) -> tuple[npt.NDArray, list[SceneObject]]:
        """Create depth map with background gradient and objects."""
        # Background gradient: far at top, near at bottom
        depth_map = self._create_background(w, h)

        # Add animated objects
        objects = self._create_objects(w, h, frame)
        self._draw_objects_on_depth(depth_map, objects)

        # Add central focus object
        center = self._create_center_object(w, h, frame)
        objects.append(center)
        self._draw_objects_on_depth(depth_map, [center])

        return depth_map, objects

    def _create_background(self, w: int, h: int) -> npt.NDArray:
        """Create depth gradient background."""
        y_ratio = np.linspace(0, 1, h).reshape(-1, 1)
        background = self.max_depth - (self.max_depth - self.min_depth * 2) * y_ratio
        return np.broadcast_to(background, (h, w)).copy().astype(np.float32)

    def _create_objects(self, w: int, h: int, frame: int) -> list[SceneObject]:
        """Create animated floating objects."""
        objects = []
        t = frame * 0.02

        for i in range(self.num_objects):
            angle = t + i * (2 * np.pi / self.num_objects)

            obj = SceneObject(
                x=int(w * (0.3 + 0.4 * np.cos(angle * 0.3 + i))),
                y=int(h * (0.3 + 0.4 * np.sin(angle * 0.5 + i * 0.7))),
                radius=int(30 + 20 * np.sin(t * 0.5 + i)),
                depth_mm=self.min_depth + i * 400 + 200 * np.sin(t + i),
            )
            objects.append(obj)

        return objects

    def _create_center_object(self, w: int, h: int, frame: int) -> SceneObject:
        """Create the central focus object."""
        t = frame * 0.02
        return SceneObject(
            x=w // 2,
            y=h // 2,
            radius=min(w, h) // 6,
            depth_mm=self.base_depth + 100 * np.sin(t),
        )

    def _draw_objects_on_depth(
        self, depth_map: npt.NDArray, objects: list[SceneObject]
    ) -> None:
        """Draw objects onto the depth map."""
        h, w = depth_map.shape
        yy, xx = np.ogrid[:h, :w]

        for obj in objects:
            mask = (xx - obj.x) ** 2 + (yy - obj.y) ** 2 <= obj.radius ** 2
            depth_map[mask] = obj.depth_mm

    def _generate_image(
        self, w: int, h: int, depth_map: npt.NDArray, frame: int
    ) -> npt.NDArray:
        """Create a colorful image based on the depth map."""
        # Normalize depth for visualization
        normalized = np.clip(
            (depth_map - self.min_depth) / (self.max_depth - self.min_depth), 0, 1
        )

        # HSV: hue from depth (red=close, green=far)
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = (normalized * 120).astype(np.uint8)
        hsv[:, :, 1] = 200
        hsv[:, :, 2] = 180 + (normalized * 75).astype(np.uint8)

        # Add animated texture
        t = frame * 0.05
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin(xx * 0.05 + t) * np.cos(yy * 0.05 + t * 0.7)
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2].astype(np.int16) + (pattern * 30).astype(np.int16), 0, 255
        ).astype(np.uint8)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Edge highlights
        edges = cv2.Canny((depth_map / 20).astype(np.uint8), 50, 150)
        bgr[edges > 0] = [255, 255, 255]

        return bgr

    def set_base_depth(self, depth_mm: float) -> None:
        """Update base depth for center object."""
        self.base_depth = depth_mm


# ============================================================================
# Stereo Synthesis
# ============================================================================


class StereoSynthesizer:
    """
    Converts a depth map + left image into a stereo pair.

    Applies per-pixel disparity based on the depth map to create
    a realistic right image.
    """

    def __init__(self, config: StereoConfig) -> None:
        self.config = config

    def synthesize_right(
        self, left: npt.NDArray, depth_map: npt.NDArray
    ) -> npt.NDArray:
        """
        Generate right frame from left frame and depth map.

        Uses the formula: disparity = focal_length * baseline / depth
        Then shifts pixels accordingly.
        """
        h, w = left.shape[:2]
        right = np.zeros_like(left)

        # Calculate disparity map
        disparity = self._compute_disparity(depth_map)

        # Apply disparity to create right image
        self._apply_disparity(left, right, disparity)

        # Fill gaps
        self._fill_gaps(right)

        return right

    def _compute_disparity(self, depth_map: npt.NDArray) -> npt.NDArray:
        """Convert depth to disparity."""
        with np.errstate(divide="ignore", invalid="ignore"):
            disparity = self.config.depth_factor / depth_map
            return np.nan_to_num(disparity, nan=0, posinf=0, neginf=0)

    def _apply_disparity(
        self, left: npt.NDArray, right: npt.NDArray, disparity: npt.NDArray
    ) -> None:
        """Shift pixels from left to right based on disparity (vectorized)."""
        h, w = left.shape[:2]
        
        # Create coordinate grids
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Compute source X coordinates
        src_x = (xx + disparity.astype(np.int32))
        
        # Valid mask: source pixels within bounds
        valid = (src_x >= 0) & (src_x < w)
        
        # Clip for safe indexing, then copy valid pixels
        src_x_clipped = np.clip(src_x, 0, w - 1)
        right[valid] = left[yy[valid], src_x_clipped[valid]]

    def _fill_gaps(self, image: npt.NDArray) -> None:
        """Fill black gaps with neighboring pixels (vectorized)."""
        h, w = image.shape[:2]
        
        # Detect gap pixels (all channels zero)
        is_gap = np.all(image == 0, axis=2)
        
        # For each row, propagate non-gap values left-to-right
        # Create a marker array: 0 where gap, column index where not gap
        marker = np.where(is_gap, 0, np.arange(w))
        
        # Use maximum.accumulate to propagate last non-gap index across each row
        fill_idx = np.maximum.accumulate(marker, axis=1)
        
        # Copy pixels from the propagated indices
        yy = np.arange(h)[:, None]
        image[:] = image[yy, fill_idx]


# ============================================================================
# Noise
# ============================================================================


class SensorNoise:
    """Simulates camera sensor noise."""

    def __init__(self, std: float = 2.0, seed: int = 42) -> None:
        self.std = std
        self._rng = np.random.default_rng(seed)

    def apply(self, image: npt.NDArray) -> npt.NDArray:
        """Add Gaussian noise to image."""
        if self.std <= 0:
            return image

        noise = self._rng.normal(0, self.std, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


# ============================================================================
# Main Camera Class
# ============================================================================


class MockStereoCamera(BaseStereoCamera):
    """
    Mock stereo camera using synthetic scene generation.

    Composes:
    - SceneGenerator: Creates depth map and left image
    - StereoSynthesizer: Creates right image from depth
    - SensorNoise: Adds realistic noise

    Example:
        >>> config = StereoConfig()
        >>> camera = MockStereoCamera(config)
        >>> left, right = camera.capture()  # Real depth variation!
    """

    __slots__ = ("_generator", "_synthesizer", "_noise", "_frame_count")

    def __init__(
        self,
        config: StereoConfig,
        camera_index: int = 0,  # Ignored, API compatibility
        simulated_depth_mm: float = 1000.0,
        add_noise: bool = True,
    ) -> None:
        super().__init__(config)

        self._generator = ProceduralSceneGenerator(base_depth_mm=simulated_depth_mm)
        self._synthesizer = StereoSynthesizer(config)
        self._noise = SensorNoise(std=2.0 if add_noise else 0.0)
        self._frame_count = 0

    @classmethod
    def create_auto(
        cls,
        camera_index: int = 0,
        baseline_mm: float = 60.0,
        simulated_depth_mm: float = 1000.0,
        add_noise: bool = True,
        prefer_resolution: Resolution | None = None,
    ) -> MockStereoCamera:
        """Create with default configuration."""
        resolution = prefer_resolution or Resolution(640, 480)
        config = StereoConfig(resolution=resolution, baseline_mm=baseline_mm)
        return cls(config=config, simulated_depth_mm=simulated_depth_mm, add_noise=add_noise)

    # ---- Properties ----

    @property
    def actual_resolution(self) -> Resolution:
        return self._config.resolution

    @property
    def actual_fps(self) -> float:
        return self._config.fps

    @property
    def simulated_depth(self) -> float:
        return self._generator.base_depth

    def set_simulated_depth(self, depth_mm: float) -> None:
        """Set base depth for center object."""
        if depth_mm <= 0:
            raise ValueError(f"Depth must be positive: {depth_mm}")
        self._generator.set_base_depth(depth_mm)

    # ---- Capture ----

    def capture(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Generate synthetic stereo pair with real depth variation."""
        self._frame_count += 1

        # Generate scene
        scene = self._generator.generate(
            self._config.width, self._config.height, self._frame_count
        )

        # Synthesize stereo pair
        left = scene.image
        right = self._synthesizer.synthesize_right(left, scene.depth_map)

        # Add noise to right image
        right = self._noise.apply(right)

        return left, right

    # ---- Lifecycle ----

    def release(self) -> None:
        """No-op for mock camera."""
        pass

    def is_opened(self) -> bool:
        return True

    def get_info(self) -> dict:
        return {
            "type": "MockStereoCamera",
            "mode": "synthetic",
            "resolution": self._config.resolution,
            "baseline_mm": self._config.baseline_mm,
        }


# Keep for backwards compatibility
NoiseMode = type("NoiseMode", (), {"NONE": 0, "GAUSSIAN": 1})
