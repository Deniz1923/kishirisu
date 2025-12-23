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
from enum import Enum, auto
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


class ObjectShape(Enum):
    """Shape types for scene objects."""

    CIRCLE = auto()
    RECTANGLE = auto()
    ELLIPSE = auto()


@dataclass(frozen=True, slots=True)
class SceneObject:
    """An object in the scene at a specific depth.
    
    Attributes:
        x, y: Center position in pixels
        width, height: Bounding dimensions
        depth_mm: Depth from camera in millimeters
        shape: Object shape type
        label: Object identifier/class
        color_hsv: Base color in HSV (hue, sat, val)
        velocity: Optional (vx, vy) in pixels/frame
    """

    x: int
    y: int
    width: int
    height: int
    depth_mm: float
    shape: ObjectShape = ObjectShape.CIRCLE
    label: str = "object"
    color_hsv: tuple[int, int, int] = (0, 200, 200)  # Red default
    velocity: tuple[float, float] = (0.0, 0.0)

    @property
    def radius(self) -> int:
        """Backward compatibility: radius for circles."""
        return min(self.width, self.height) // 2

    def __str__(self) -> str:
        return f"{self.label}({self.shape.name}) @ ({self.x},{self.y}) depth={self.depth_mm:.0f}mm"


@dataclass(slots=True)
class SceneInfo:
    """Metadata about a generated scene.
    
    Attributes:
        frame_number: Current frame counter
        object_count: Number of objects in scene
        depth_range: (min_depth, max_depth) in mm
        has_occlusion: Whether objects overlap
        complexity: Scene complexity score (1-10)
        description: Human-readable scene description
    """

    frame_number: int
    object_count: int
    depth_range: tuple[float, float]
    has_occlusion: bool
    complexity: int
    description: str

    def format_verbose(self) -> str:
        """Format as multi-line verbose string."""
        return (
            f"  Frame: {self.frame_number} | Objects: {self.object_count}\n"
            f"  Depth: {self.depth_range[0]:.0f}-{self.depth_range[1]:.0f}mm\n"
            f"  Occlusion: {'Yes' if self.has_occlusion else 'No'} | Complexity: {self.complexity}/10\n"
            f"  {self.description}"
        )


@dataclass(slots=True)
class Scene:
    """A complete scene with depth map, image, and metadata."""

    depth_map: npt.NDArray
    image: npt.NDArray
    objects: list[SceneObject] = field(default_factory=list)
    info: SceneInfo | None = None


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
    Generates procedural scenes with varying depth and object variety.

    Creates:
    - Floor plane with depth gradient
    - Back wall at maximum depth
    - Multiple object shapes (circles, rectangles, ellipses)
    - Central focus object (target ball)
    - Rich scene metadata
    """

    # Object labels and their properties
    OBJECT_TYPES = [
        ("ball", ObjectShape.CIRCLE, (0, 200, 230)),      # Red
        ("box", ObjectShape.RECTANGLE, (100, 180, 200)),   # Green
        ("obstacle", ObjectShape.ELLIPSE, (30, 200, 220)), # Orange
        ("target", ObjectShape.CIRCLE, (120, 255, 255)),   # Cyan
    ]

    def __init__(
        self,
        base_depth_mm: float = 1000.0,
        min_depth_mm: float = 300.0,
        max_depth_mm: float = 5000.0,
        num_objects: int = 8,
        seed: int = 42,
    ) -> None:
        self.base_depth = base_depth_mm
        self.min_depth = min_depth_mm
        self.max_depth = max_depth_mm
        self.num_objects = num_objects
        self._rng = np.random.default_rng(seed)
        self._frame_count = 0

    def generate(self, width: int, height: int, frame: int) -> Scene:
        """Generate a complete scene with metadata."""
        self._frame_count = frame
        depth_map, objects = self._generate_depth_map(width, height, frame)
        image = self._generate_image(width, height, depth_map, objects, frame)
        
        # Generate scene metadata
        info = self._generate_scene_info(frame, objects, depth_map)
        
        return Scene(depth_map=depth_map, image=image, objects=objects, info=info)

    def _generate_scene_info(
        self, frame: int, objects: list[SceneObject], depth_map: npt.NDArray
    ) -> SceneInfo:
        """Generate verbose scene metadata."""
        valid_depths = depth_map[depth_map > 0]
        min_d = float(np.min(valid_depths)) if len(valid_depths) > 0 else 0
        max_d = float(np.max(valid_depths)) if len(valid_depths) > 0 else 0
        
        # Check for occlusion (overlapping objects)
        has_occlusion = self._check_occlusion(objects)
        
        # Compute complexity (based on object count and variety)
        shape_variety = len(set(obj.shape for obj in objects))
        complexity = min(10, len(objects) + shape_variety)
        
        # Generate description
        shapes = [obj.shape.name.lower() for obj in objects]
        shape_counts = {s: shapes.count(s) for s in set(shapes)}
        desc_parts = [f"{count} {shape}{'s' if count > 1 else ''}" 
                      for shape, count in sorted(shape_counts.items())]
        description = f"Scene with {', '.join(desc_parts)}"
        
        return SceneInfo(
            frame_number=frame,
            object_count=len(objects),
            depth_range=(min_d, max_d),
            has_occlusion=has_occlusion,
            complexity=complexity,
            description=description,
        )

    def _check_occlusion(self, objects: list[SceneObject]) -> bool:
        """Check if any objects overlap."""
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1:]:
                dx = abs(obj1.x - obj2.x)
                dy = abs(obj1.y - obj2.y)
                min_dist = (obj1.radius + obj2.radius) * 0.8
                if dx < min_dist and dy < min_dist:
                    return True
        return False

    def _generate_depth_map(
        self, w: int, h: int, frame: int
    ) -> tuple[npt.NDArray, list[SceneObject]]:
        """Create depth map with floor, back wall, and objects."""
        # Create floor/wall background
        depth_map = self._create_background(w, h)

        # Add diverse animated objects
        objects = self._create_diverse_objects(w, h, frame)
        self._draw_objects_on_depth(depth_map, objects)

        # Add central focus object (the target ball)
        center = self._create_center_object(w, h, frame)
        objects.append(center)
        self._draw_objects_on_depth(depth_map, [center])

        return depth_map, objects

    def _create_background(self, w: int, h: int) -> npt.NDArray:
        """Create floor plane with depth gradient and back wall."""
        # Floor: near at bottom, far at top (perspective)
        y_ratio = np.linspace(0, 1, h).reshape(-1, 1)
        floor = self.max_depth - (self.max_depth - self.min_depth * 2) * y_ratio
        
        depth_map = np.broadcast_to(floor, (h, w)).copy().astype(np.float32)
        
        # Add back wall at top portion
        wall_height = h // 4
        depth_map[:wall_height, :] = self.max_depth
        
        return depth_map

    def _create_diverse_objects(self, w: int, h: int, frame: int) -> list[SceneObject]:
        """Create animated objects with diverse shapes and properties."""
        objects = []
        t = frame * 0.02

        for i in range(self.num_objects):
            angle = t + i * (2 * np.pi / self.num_objects)
            
            # Select object type based on index
            obj_type = self.OBJECT_TYPES[i % len(self.OBJECT_TYPES)]
            label, shape, color = obj_type

            # Animated position
            x = int(w * (0.2 + 0.6 * (0.5 + 0.4 * np.cos(angle * 0.3 + i))))
            y = int(h * (0.25 + 0.5 * (0.5 + 0.4 * np.sin(angle * 0.5 + i * 0.7))))
            
            # Animated size with shape-dependent scaling
            base_size = 40 + 30 * np.sin(t * 0.5 + i)
            if shape == ObjectShape.RECTANGLE:
                width = int(base_size * 1.5)
                height = int(base_size * 0.8)
            elif shape == ObjectShape.ELLIPSE:
                width = int(base_size * 1.2)
                height = int(base_size)
            else:  # CIRCLE
                width = height = int(base_size)
            
            # Depth varies per object
            depth = self.min_depth + (i * 350) + 200 * np.sin(t + i)
            
            # Velocity for motion blur / prediction
            vx = float(2 * np.cos(angle))
            vy = float(1.5 * np.sin(angle * 0.7))

            obj = SceneObject(
                x=x,
                y=y,
                width=width,
                height=height,
                depth_mm=depth,
                shape=shape,
                label=f"{label}_{i}",
                color_hsv=color,
                velocity=(vx, vy),
            )
            objects.append(obj)

        return objects

    def _create_center_object(self, w: int, h: int, frame: int) -> SceneObject:
        """Create the central target ball."""
        t = frame * 0.02
        size = min(w, h) // 6
        return SceneObject(
            x=w // 2,
            y=h // 2,
            width=size,
            height=size,
            depth_mm=self.base_depth + 100 * np.sin(t),
            shape=ObjectShape.CIRCLE,
            label="target_center",
            color_hsv=(0, 255, 255),  # Bright red
            velocity=(0.0, 0.0),
        )

    def _draw_objects_on_depth(
        self, depth_map: npt.NDArray, objects: list[SceneObject]
    ) -> None:
        """Draw objects onto the depth map based on their shape."""
        h, w = depth_map.shape
        yy, xx = np.ogrid[:h, :w]

        for obj in objects:
            if obj.shape == ObjectShape.CIRCLE:
                r = obj.radius
                mask = (xx - obj.x) ** 2 + (yy - obj.y) ** 2 <= r ** 2
            elif obj.shape == ObjectShape.RECTANGLE:
                half_w, half_h = obj.width // 2, obj.height // 2
                mask = (
                    (xx >= obj.x - half_w) & (xx <= obj.x + half_w) &
                    (yy >= obj.y - half_h) & (yy <= obj.y + half_h)
                )
            elif obj.shape == ObjectShape.ELLIPSE:
                # Ellipse equation: (x-cx)^2/a^2 + (y-cy)^2/b^2 <= 1
                a, b = obj.width / 2, obj.height / 2
                mask = ((xx - obj.x) / a) ** 2 + ((yy - obj.y) / b) ** 2 <= 1
            else:
                mask = np.zeros((h, w), dtype=bool)
            
            depth_map[mask] = obj.depth_mm

    def _generate_image(
        self, w: int, h: int, depth_map: npt.NDArray, objects: list[SceneObject], frame: int
    ) -> npt.NDArray:
        """Create a detailed colorful image based on the scene."""
        # Base colors from depth
        normalized = np.clip(
            (depth_map - self.min_depth) / (self.max_depth - self.min_depth), 0, 1
        )

        # HSV: hue from depth (red=close, green=far)
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = (normalized * 120).astype(np.uint8)
        hsv[:, :, 1] = 180
        hsv[:, :, 2] = 160 + (normalized * 60).astype(np.uint8)

        # Add animated texture pattern (checkerboard-ish)
        t = frame * 0.05
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin(xx * 0.08 + t) * np.cos(yy * 0.08 + t * 0.7)
        pattern += np.sin(xx * 0.03 - t * 0.5) * np.sin(yy * 0.03 + t * 0.3) * 0.5
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2].astype(np.int16) + (pattern * 25).astype(np.int16), 0, 255
        ).astype(np.uint8)

        # Draw objects with their individual colors
        for obj in objects:
            self._draw_object_color(hsv, obj)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Edge highlights for depth discontinuities
        edges = cv2.Canny((depth_map / 20).astype(np.uint8), 50, 150)
        bgr[edges > 0] = [255, 255, 255]

        return bgr

    def _draw_object_color(self, hsv: npt.NDArray, obj: SceneObject) -> None:
        """Draw object's color on the HSV image."""
        h, w = hsv.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        
        if obj.shape == ObjectShape.CIRCLE:
            r = obj.radius
            mask = (xx - obj.x) ** 2 + (yy - obj.y) ** 2 <= r ** 2
        elif obj.shape == ObjectShape.RECTANGLE:
            half_w, half_h = obj.width // 2, obj.height // 2
            mask = (
                (xx >= obj.x - half_w) & (xx <= obj.x + half_w) &
                (yy >= obj.y - half_h) & (yy <= obj.y + half_h)
            )
        else:  # ELLIPSE
            a, b = obj.width / 2, obj.height / 2
            mask = ((xx - obj.x) / max(a, 1)) ** 2 + ((yy - obj.y) / max(b, 1)) ** 2 <= 1
        
        hsv[:, :, 0][mask] = obj.color_hsv[0]
        hsv[:, :, 1][mask] = obj.color_hsv[1]
        hsv[:, :, 2][mask] = obj.color_hsv[2]

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
