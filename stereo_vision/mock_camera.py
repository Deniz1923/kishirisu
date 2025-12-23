"""
Mock Stereo Camera
===================

Generates synthetic stereo pairs from a single webcam for development/testing.

How It Works:
- Captures a single frame from webcam → "left" image
- Creates synthetic "right" image by horizontal translation
- The shift amount encodes simulated depth via disparity formula

Limitations:
- Depth is simulated, not measured
- No real parallax/occlusion effects
- Useful for algorithm development, NOT actual measurements
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .capture import BaseStereoCamera
from .config import Resolution, StereoConfig

if TYPE_CHECKING:
    import numpy.typing as npt


class NoiseMode(Enum):
    """Sensor noise simulation modes."""

    NONE = auto()
    GAUSSIAN = auto()
    SALT_PEPPER = auto()


@dataclass(slots=True)
class MockCameraState:
    """Mutable state for mock camera."""

    simulated_depth_mm: float = 1000.0
    noise_mode: NoiseMode = NoiseMode.GAUSSIAN
    noise_std: float = 3.0


class MockStereoCamera(BaseStereoCamera):
    """
    Mock stereo camera using a single webcam.

    Generates synthetic stereo pairs by applying horizontal shifts
    to simulate what a second camera would see.

    Example:
        >>> config = StereoConfig.from_camera()
        >>> with MockStereoCamera(config) as camera:
        ...     left, right = camera.capture()
        ...     camera.set_simulated_depth(500)  # Closer object
        ...     left2, right2 = camera.capture()
    """

    __slots__ = ("_cap", "_state", "_actual_resolution", "_actual_fps")

    def __init__(
        self,
        config: StereoConfig,
        camera_index: int = 0,
        simulated_depth_mm: float = 1000.0,
        add_noise: bool = True,
    ) -> None:
        """
        Initialize mock stereo camera.

        Args:
            config: Stereo configuration
            camera_index: OpenCV camera index (0 = default)
            simulated_depth_mm: Initial simulated depth in mm
            add_noise: Whether to add sensor noise to right image
        """
        super().__init__(config)

        self._state = MockCameraState(
            simulated_depth_mm=simulated_depth_mm,
            noise_mode=NoiseMode.GAUSSIAN if add_noise else NoiseMode.NONE,
        )

        self._cap: cv2.VideoCapture | None = None
        self._actual_resolution: Resolution | None = None
        self._actual_fps: float = 0.0

        self._initialize_camera(camera_index)

    def _initialize_camera(self, camera_index: int) -> None:
        """Initialize webcam capture."""
        self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {camera_index}. "
                "Ensure webcam is connected and not in use."
            )

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

        if self._config.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        # Read actual values
        self._actual_resolution = Resolution(
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @classmethod
    def create_auto(
        cls,
        camera_index: int = 0,
        baseline_mm: float = 60.0,
        simulated_depth_mm: float = 1000.0,
        add_noise: bool = True,
        prefer_resolution: Resolution | None = None,
    ) -> MockStereoCamera:
        """
        Create MockStereoCamera with auto-detected configuration.

        Args:
            camera_index: Camera to use
            baseline_mm: Stereo baseline in mm
            simulated_depth_mm: Initial simulated depth
            add_noise: Add sensor noise simulation
            prefer_resolution: Preferred resolution (if available)

        Returns:
            Configured MockStereoCamera instance
        """
        config = StereoConfig.from_camera(
            camera_index=camera_index,
            baseline_mm=baseline_mm,
            prefer_resolution=prefer_resolution,
        )
        return cls(
            config=config,
            camera_index=camera_index,
            simulated_depth_mm=simulated_depth_mm,
            add_noise=add_noise,
        )

    @property
    def actual_resolution(self) -> Resolution | None:
        """Get actual camera resolution."""
        return self._actual_resolution

    @property
    def actual_fps(self) -> float:
        """Get actual camera FPS."""
        return self._actual_fps

    @property
    def simulated_depth(self) -> float:
        """Get current simulated depth in mm."""
        return self._state.simulated_depth_mm

    def set_simulated_depth(self, depth_mm: float) -> None:
        """
        Set simulated scene depth.

        Args:
            depth_mm: Depth in millimeters (must be positive)

        Raises:
            ValueError: If depth is not positive
        """
        if depth_mm <= 0:
            raise ValueError(f"Depth must be positive: {depth_mm}")
        self._state.simulated_depth_mm = depth_mm

    def capture(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Capture a stereo pair.

        The left image is the raw webcam frame.
        The right image is synthetically shifted based on simulated depth.

        Returns:
            (left_frame, right_frame) as BGR numpy arrays

        Raises:
            RuntimeError: If capture fails
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not initialized")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        # Resize if needed
        h, w = frame.shape[:2]
        if w != self._config.width or h != self._config.height:
            frame = cv2.resize(frame, (self._config.width, self._config.height))

        left_frame = frame.copy()
        right_frame = self._generate_right_frame(frame)

        return left_frame, right_frame

    def _generate_right_frame(self, left_frame: npt.NDArray) -> npt.NDArray:
        """Generate synthetic right camera image."""
        h, w = left_frame.shape[:2]

        # Calculate disparity: d = (f * B) / Z
        disparity = self._config.depth_factor / self._state.simulated_depth_mm

        # Create horizontal shift matrix
        # In stereo: right camera sees objects shifted LEFT
        shift_matrix = np.array(
            [[1.0, 0.0, -disparity], [0.0, 1.0, 0.0]], dtype=np.float32
        )

        right_frame = cv2.warpAffine(
            left_frame, shift_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

        # Add sensor noise
        if self._state.noise_mode == NoiseMode.GAUSSIAN:
            right_frame = self._add_gaussian_noise(right_frame)

        return right_frame

    def _add_gaussian_noise(self, frame: npt.NDArray) -> npt.NDArray:
        """Add Gaussian sensor noise."""
        noise = np.random.normal(0, self._state.noise_std, frame.shape)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def release(self) -> None:
        """Release webcam resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def get_info(self) -> dict:
        """Get camera information."""
        return {
            "actual_resolution": self._actual_resolution,
            "actual_fps": self._actual_fps,
            "config_resolution": self._config.resolution,
            "baseline_mm": self._config.baseline_mm,
            "simulated_depth_mm": self._state.simulated_depth_mm,
        }
