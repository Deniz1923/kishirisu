"""
Real Stereo Camera Capture
===========================

Provides real stereo camera capture from hardware sources.

Supports:
- Single camera with side-by-side stereo (split mode)
- Dual camera setup (two separate cameras)

Example:
    >>> from stereo_vision import RealStereoCamera, StereoConfig
    >>> config = StereoConfig.from_camera(camera_index=0)
    >>> with RealStereoCamera(config, camera_index=0) as camera:
    ...     left, right = camera.capture()
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .capture import BaseStereoCamera

if TYPE_CHECKING:
    import numpy.typing as npt
    from .config import StereoConfig


__all__ = [
    "CaptureMode",
    "RealStereoCamera",
]


class CaptureMode(Enum):
    """Stereo capture mode for real cameras."""

    SPLIT = auto()  # Single camera, side-by-side stereo image
    DUAL = auto()  # Two separate cameras


class RealStereoCamera(BaseStereoCamera):
    """
    Real stereo camera capture from hardware.

    Supports two modes:
    - SPLIT: Single camera with side-by-side stereo image (e.g., stereo lens adapter)
    - DUAL: Two separate USB cameras

    Example:
        >>> config = StereoConfig.from_camera(camera_index=0)
        >>> camera = RealStereoCamera(config, camera_index=0, mode=CaptureMode.SPLIT)
        >>> left, right = camera.capture()
        >>> camera.release()

    Attributes:
        mode: The capture mode (SPLIT or DUAL)
        is_opened: Whether the camera(s) are successfully opened
    """

    __slots__ = ("_cap_left", "_cap_right", "_mode", "_camera_index", "_opened")

    def __init__(
        self,
        config: StereoConfig,
        camera_index: int = 0,
        mode: CaptureMode = CaptureMode.SPLIT,
        right_camera_index: int | None = None,
    ) -> None:
        """
        Initialize real stereo camera.

        Args:
            config: Stereo configuration
            camera_index: Primary camera index (or left camera for DUAL mode)
            mode: Capture mode (SPLIT or DUAL)
            right_camera_index: Right camera index for DUAL mode (default: camera_index + 1)
        """
        super().__init__(config)
        self._mode = mode
        self._camera_index = camera_index
        self._opened = False

        # Initialize primary/left camera
        self._cap_left = cv2.VideoCapture(camera_index)
        self._cap_right: cv2.VideoCapture | None = None

        if not self._cap_left.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")

        # Configure resolution
        self._cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, config.width * (2 if mode == CaptureMode.SPLIT else 1))
        self._cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

        # For DUAL mode, open second camera
        if mode == CaptureMode.DUAL:
            right_idx = right_camera_index if right_camera_index is not None else camera_index + 1
            self._cap_right = cv2.VideoCapture(right_idx)
            if not self._cap_right.isOpened():
                self._cap_left.release()
                raise RuntimeError(f"Failed to open right camera {right_idx}")

            self._cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            self._cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

        self._opened = True

    @classmethod
    def create_auto(
        cls,
        camera_index: int = 0,
        baseline_mm: float = 60.0,
        mode: CaptureMode = CaptureMode.SPLIT,
        right_camera_index: int | None = None,
    ) -> RealStereoCamera:
        """
        Create with automatic configuration detection.

        Args:
            camera_index: Primary camera index
            baseline_mm: Stereo baseline in mm
            mode: Capture mode
            right_camera_index: Right camera index for DUAL mode

        Returns:
            Configured RealStereoCamera instance
        """
        from .config import StereoConfig

        config = StereoConfig.from_camera(
            camera_index=camera_index,
            baseline_mm=baseline_mm,
        )
        return cls(config, camera_index, mode, right_camera_index)

    @property
    def mode(self) -> CaptureMode:
        """Get the capture mode."""
        return self._mode

    @property
    def is_opened(self) -> bool:
        """Check if camera(s) are opened."""
        return self._opened

    def capture(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Capture a stereo pair from hardware.

        Returns:
            Tuple of (left_frame, right_frame) as BGR numpy arrays
            with shape (height, width, 3).

        Raises:
            RuntimeError: If capture fails or camera not opened
        """
        if not self._opened:
            raise RuntimeError("Camera not opened")

        if self._mode == CaptureMode.SPLIT:
            return self._capture_split()
        else:
            return self._capture_dual()

    def _capture_split(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Capture from single side-by-side stereo camera."""
        ret, frame = self._cap_left.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera")

        # Split the frame horizontally
        h, w = frame.shape[:2]
        mid = w // 2

        left = frame[:, :mid].copy()
        right = frame[:, mid:].copy()

        # Resize if needed to match config
        target_w, target_h = self._config.width, self._config.height
        if left.shape[:2] != (target_h, target_w):
            left = cv2.resize(left, (target_w, target_h))
            right = cv2.resize(right, (target_w, target_h))

        return left, right

    def _capture_dual(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Capture from two separate cameras."""
        if self._cap_right is None:
            raise RuntimeError("Right camera not initialized for DUAL mode")

        ret_l, left = self._cap_left.read()
        ret_r, right = self._cap_right.read()

        if not ret_l or left is None:
            raise RuntimeError("Failed to capture from left camera")
        if not ret_r or right is None:
            raise RuntimeError("Failed to capture from right camera")

        # Resize if needed
        target_w, target_h = self._config.width, self._config.height
        if left.shape[:2] != (target_h, target_w):
            left = cv2.resize(left, (target_w, target_h))
        if right.shape[:2] != (target_h, target_w):
            right = cv2.resize(right, (target_w, target_h))

        return left, right

    def release(self) -> None:
        """Release camera resources."""
        if self._cap_left is not None:
            self._cap_left.release()
        if self._cap_right is not None:
            self._cap_right.release()
        self._opened = False

    def get_info(self) -> str:
        """Get camera info string."""
        mode_str = "Split" if self._mode == CaptureMode.SPLIT else "Dual"
        return (
            f"RealStereoCamera(mode={mode_str}, "
            f"index={self._camera_index}, "
            f"resolution={self._config.width}x{self._config.height})"
        )
