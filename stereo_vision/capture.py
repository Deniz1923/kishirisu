"""
Stereo Camera Capture Interface
================================

This module defines the abstract base class for stereo camera capture.
All camera implementations (mock, real hardware) must inherit from this.

The abstraction allows swapping between:
- MockStereoCamera: Uses single webcam for development
- RealStereoCamera: Uses actual dual cameras on the robot (future)
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from .config import StereoConfig


class StereoCapture(ABC):
    """
    Abstract base class for stereo camera capture.
    
    This interface defines what any stereo camera implementation must provide.
    By coding to this interface, you can easily swap between mock and real cameras.
    
    Example:
        >>> camera: StereoCapture = MockStereoCamera(config)  # For development
        >>> # Later, swap to:
        >>> # camera: StereoCapture = RealStereoCamera(config)  # For robot
        >>> 
        >>> left, right = camera.capture()
        >>> process_stereo_pair(left, right)
    
    Attributes:
        config: StereoConfig instance with camera parameters
        current_x: Current X-axis position of the camera mount
        current_y: Current Y-axis position of the camera mount
    """
    
    def __init__(self, config: StereoConfig):
        """
        Initialize the stereo capture with configuration.
        
        Args:
            config: StereoConfig instance containing camera parameters
        """
        self._config = config
        # Camera position on movable mount (starts at center)
        self._current_x: float = 0.0
        self._current_y: float = 0.0
    
    @property
    def config(self) -> StereoConfig:
        """Get the stereo configuration."""
        return self._config
    
    @property
    def current_x(self) -> float:
        """Get current X-axis position in mm."""
        return self._current_x
    
    @property
    def current_y(self) -> float:
        """Get current Y-axis position in mm."""
        return self._current_y
    
    @abstractmethod
    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture a stereo pair of images.
        
        This method must be implemented by all subclasses.
        
        Returns:
            Tuple of (left_frame, right_frame) where each frame is a
            numpy array of shape (height, width, 3) in BGR format.
            
        Raises:
            RuntimeError: If camera capture fails
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """
        Release camera resources.
        
        Should be called when done using the camera to free resources.
        """
        pass
    
    def set_position(self, x: float, y: float) -> None:
        """
        Set the camera mount position on X/Y axes.
        
        The cameras on your robot are mounted on a movable platform controlled
        by servo motors. This method simulates or controls that movement.
        
        Args:
            x: Target X position in mm (will be clamped to valid range)
            y: Target Y position in mm (will be clamped to valid range)
            
        Note:
            Values are clamped to the valid range defined in StereoConfig.
            Check current_x and current_y properties after calling to see
            the actual position.
        """
        # Clamp X to valid range
        x_min, x_max = self._config.x_axis_range
        self._current_x = max(x_min, min(x_max, x))
        
        # Clamp Y to valid range
        y_min, y_max = self._config.y_axis_range
        self._current_y = max(y_min, min(y_max, y))
    
    def move_relative(self, dx: float, dy: float) -> None:
        """
        Move camera position relative to current position.
        
        Args:
            dx: Delta X in mm (positive = right, negative = left)
            dy: Delta Y in mm (positive = up, negative = down)
        """
        self.set_position(self._current_x + dx, self._current_y + dy)
    
    def reset_position(self) -> None:
        """Reset camera to center position (0, 0)."""
        self.set_position(0.0, 0.0)
    
    def __enter__(self):
        """Context manager entry - allows 'with' statement usage."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are released."""
        self.release()
        return False
