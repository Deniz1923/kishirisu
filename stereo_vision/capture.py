"""
Stereo Camera Capture Interface
================================

Protocol-based interface for stereo camera capture sources.
Supports both mock cameras (for development) and real hardware.

This module provides:
- StereoCapture: Protocol defining the capture interface
- CameraPosition: Immutable position state for movable mounts
- BaseStereoCamera: Optional base class with common functionality
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy.typing as npt
    from .config import StereoConfig


__all__ = [
    "CameraPosition",
    "StereoCapture",
    "BaseStereoCamera",
]


@dataclass(frozen=True, slots=True)
class CameraPosition:
    """Immutable camera mount position in millimeters."""

    x: float = 0.0
    y: float = 0.0

    def moved(self, dx: float = 0.0, dy: float = 0.0) -> CameraPosition:
        """Return new position offset by (dx, dy)."""
        return CameraPosition(self.x + dx, self.y + dy)

    def clamped(
        self,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-30.0, 30.0),
    ) -> CameraPosition:
        """Return position clamped to valid ranges."""
        return CameraPosition(
            x=max(x_range[0], min(x_range[1], self.x)),
            y=max(y_range[0], min(y_range[1], self.y)),
        )

    def __str__(self) -> str:
        return f"({self.x:+.1f}, {self.y:+.1f})mm"


@runtime_checkable
class StereoCapture(Protocol):
    """
    Protocol for stereo camera capture sources.

    Implementations must provide:
    - config: Access to stereo configuration
    - capture(): Return (left, right) image pair
    - release(): Free camera resources

    Example:
        >>> camera: StereoCapture = MockStereoCamera(config)
        >>> left, right = camera.capture()
        >>> camera.release()

        # Or use as context manager:
        >>> with MockStereoCamera(config) as camera:
        ...     left, right = camera.capture()
    """

    @property
    def config(self) -> StereoConfig:
        """Get the stereo configuration."""
        ...

    def capture(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Capture a stereo pair of images.

        Returns:
            Tuple of (left_frame, right_frame) as BGR numpy arrays
            with shape (height, width, 3).

        Raises:
            RuntimeError: If capture fails
        """
        ...

    def release(self) -> None:
        """Release camera resources."""
        ...


class BaseStereoCamera(AbstractContextManager):
    """
    Base class providing common stereo camera functionality.

    Concrete implementations should subclass this and implement:
    - capture(): The actual frame capture logic
    - release(): Resource cleanup

    Provides:
    - Context manager support (with statement)
    - Camera position tracking for movable mounts
    - Position clamping to valid ranges
    """

    __slots__ = ("_config", "_position", "_x_range", "_y_range")

    def __init__(
        self,
        config: StereoConfig,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-30.0, 30.0),
    ) -> None:
        """
        Initialize base stereo camera.

        Args:
            config: Stereo configuration
            x_range: Valid X-axis movement range in mm
            y_range: Valid Y-axis movement range in mm
        """
        self._config = config
        self._position = CameraPosition(0.0, 0.0)
        self._x_range = x_range
        self._y_range = y_range

    @property
    def config(self) -> StereoConfig:
        """Get the stereo configuration."""
        return self._config

    @property
    def position(self) -> CameraPosition:
        """Get current camera mount position."""
        return self._position

    @property
    def current_x(self) -> float:
        """Get current X-axis position in mm."""
        return self._position.x

    @property
    def current_y(self) -> float:
        """Get current Y-axis position in mm."""
        return self._position.y

    def set_position(self, x: float, y: float) -> None:
        """
        Set camera mount position, clamping to valid range.

        Args:
            x: Target X position in mm
            y: Target Y position in mm
        """
        self._position = CameraPosition(x, y).clamped(self._x_range, self._y_range)

    def move_relative(self, dx: float, dy: float) -> None:
        """
        Move camera position relative to current.

        Args:
            dx: Delta X in mm (positive = right)
            dy: Delta Y in mm (positive = down)
        """
        new_pos = self._position.moved(dx, dy).clamped(self._x_range, self._y_range)
        self._position = new_pos

    def reset_position(self) -> None:
        """Reset camera to center position (0, 0)."""
        self._position = CameraPosition(0.0, 0.0)

    def capture(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Capture stereo pair - must be implemented by subclass."""
        raise NotImplementedError

    def release(self) -> None:
        """Release resources - must be implemented by subclass."""
        raise NotImplementedError

    def __enter__(self) -> BaseStereoCamera:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - ensures release is called."""
        self.release()
        return False
