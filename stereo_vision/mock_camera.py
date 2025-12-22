"""
Mock Stereo Camera
===================

This module provides a mock stereo camera implementation that uses a single
laptop webcam to generate synthetic stereo image pairs.

How Mock Stereo Works:
---------------------
Real stereo cameras capture two slightly different views of the same scene.
The difference (disparity) between matching points in left/right images
encodes depth information.

Since you only have ONE camera, we simulate the second camera by:
1. Capturing a single frame from the webcam (this becomes the "left" image)
2. Creating a synthetic "right" image by shifting pixels horizontally
3. The shift amount corresponds to simulated depth - closer objects shift more

This allows you to develop and test your depth pipeline before getting
the actual stereo hardware.

Limitations:
-----------
- Depth is simulated, not measured
- No real parallax effects (objects don't reveal hidden surfaces)
- Useful for algorithm development, NOT for actual measurements
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from .capture import StereoCapture
from .config import StereoConfig


class MockStereoCamera(StereoCapture):
    """
    Mock stereo camera using a single webcam.
    
    This class captures from your laptop camera and generates a synthetic
    stereo pair by applying horizontal shifts and perspective transforms.
    
    The synthetic "right" image is created by:
    1. Horizontal translation (simulating camera baseline)
    2. Optional perspective warp (simulating viewing angle difference)
    3. Optional noise addition (simulating camera sensor differences)
    
    Example:
        >>> # Auto-detect camera capabilities
        >>> config = StereoConfig.from_camera(camera_index=0)
        >>> camera = MockStereoCamera(config, camera_index=0)
        >>> print(f"Using: {camera.actual_resolution} @ {camera.actual_fps} FPS")
        >>> 
        >>> # Capture stereo pair
        >>> left, right = camera.capture()
        >>> camera.release()
        
        >>> # Or use auto-detection directly
        >>> camera = MockStereoCamera.create_auto(camera_index=0)
        >>> left, right = camera.capture()
    
    Args:
        config: StereoConfig with camera parameters
        camera_index: OpenCV camera index (0 = default webcam)
        simulated_depth_mm: Simulated scene depth for disparity calculation
        add_noise: Whether to add realistic noise to the right image
        auto_update_config: If True, update config resolution to match actual camera
    """
    
    def __init__(
        self,
        config: StereoConfig,
        camera_index: int = 0,
        simulated_depth_mm: float = 1000.0,
        add_noise: bool = True,
        auto_update_config: bool = True
    ):
        """
        Initialize mock stereo camera.
        
        Args:
            config: Stereo configuration parameters
            camera_index: Which camera to use (0 = default webcam)
            simulated_depth_mm: Simulated depth of the scene in mm.
                              This controls how much the right image is shifted.
                              Closer = more shift, farther = less shift.
            add_noise: If True, adds slight gaussian noise to right image
                      to simulate real camera sensor differences.
            auto_update_config: If True and camera provides different resolution
                               than configured, create updated config automatically.
        """
        super().__init__(config)
        
        self._camera_index = camera_index
        self._simulated_depth_mm = simulated_depth_mm
        self._add_noise = add_noise
        self._auto_update_config = auto_update_config
        
        # These will be populated after initialization
        self._actual_resolution: Optional[Tuple[int, int]] = None
        self._actual_fps: Optional[float] = None
        
        # Initialize the webcam capture
        # cv2.VideoCapture opens a connection to the camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._initialize_camera()
    
    @classmethod
    def create_auto(
        cls,
        camera_index: int = 0,
        baseline_mm: float = 60.0,
        simulated_depth_mm: float = 1000.0,
        add_noise: bool = True,
        prefer_resolution: Optional[Tuple[int, int]] = None
    ) -> "MockStereoCamera":
        """
        Create a MockStereoCamera with automatic camera detection.
        
        This is the easiest way to get started - it detects your camera's
        native resolution and FPS automatically.
        
        Args:
            camera_index: Which camera to use (0 = default webcam)
            baseline_mm: Stereo baseline in mm (measure your actual cameras)
            simulated_depth_mm: Initial simulated depth
            add_noise: Whether to add sensor noise simulation
            prefer_resolution: If set, try to use this resolution instead of max
            
        Returns:
            MockStereoCamera configured for your camera
            
        Example:
            >>> camera = MockStereoCamera.create_auto(camera_index=0)
            >>> print(f"Detected: {camera.actual_resolution} @ {camera.actual_fps} FPS")
            >>> left, right = camera.capture()
        """
        # Auto-detect camera capabilities
        config = StereoConfig.from_camera(
            camera_index=camera_index,
            baseline_mm=baseline_mm,
            prefer_resolution=prefer_resolution
        )
        
        return cls(
            config=config,
            camera_index=camera_index,
            simulated_depth_mm=simulated_depth_mm,
            add_noise=add_noise,
            auto_update_config=False  # Already detected
        )
    
    def _initialize_camera(self) -> None:
        """
        Initialize the webcam capture.
        
        Detects actual camera resolution and FPS.
        Updates config if auto_update_config is enabled.
        Raises RuntimeError if camera cannot be opened.
        """
        # Open camera connection
        self._cap = cv2.VideoCapture(self._camera_index)
        
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera at index {self._camera_index}. "
                "Make sure your webcam is connected and not in use."
            )
        
        # Try to set camera to configured resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        
        # Also try to set FPS if specified
        if self._config.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
        
        # Read what the camera actually provides
        self._actual_resolution = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self._actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._actual_fps <= 0:
            self._actual_fps = 30.0  # Fallback
        
        # Report what we got
        if self._actual_resolution != self._config.resolution:
            print(f"[MockStereoCamera] Requested: {self._config.resolution}, "
                  f"Got: {self._actual_resolution}")
            
            if self._auto_update_config:
                # Update config to match actual camera
                print(f"[MockStereoCamera] Auto-updating config to match camera")
                self._config = self._config.scale_to_resolution(self._actual_resolution)
        else:
            print(f"[MockStereoCamera] Resolution: {self._actual_resolution[0]}x{self._actual_resolution[1]}")
        
        print(f"[MockStereoCamera] FPS: {self._actual_fps}")
    
    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """Get the actual camera resolution being used."""
        return self._actual_resolution
    
    @property
    def actual_fps(self) -> float:
        """Get the actual camera FPS."""
        return self._actual_fps
    
    @property
    def actual_width(self) -> int:
        """Get actual frame width in pixels."""
        return self._actual_resolution[0]
    
    @property
    def actual_height(self) -> int:
        """Get actual frame height in pixels."""
        return self._actual_resolution[1]
    
    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture a stereo pair from the webcam.
        
        The left image is the raw webcam frame.
        The right image is synthetically generated by shifting the left image
        horizontally by an amount that corresponds to the simulated depth.
        
        The disparity formula is:
            disparity = (focal_length * baseline) / depth
            
        So closer objects (small depth) have larger disparity (more shift).
        
        Returns:
            Tuple of (left_frame, right_frame) as BGR numpy arrays
            
        Raises:
            RuntimeError: If frame capture fails
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not initialized")
        
        # Capture frame from webcam
        ret, frame = self._cap.read()
        
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        
        # Resize if camera gives different resolution than expected
        if frame.shape[1] != self._config.width or frame.shape[0] != self._config.height:
            frame = cv2.resize(frame, (self._config.width, self._config.height))
        
        # The captured frame becomes our "left" camera image
        left_frame = frame.copy()
        
        # Generate synthetic "right" camera image
        right_frame = self._generate_right_image(frame)
        
        return left_frame, right_frame
    
    def _generate_right_image(self, left_frame: np.ndarray) -> np.ndarray:
        """
        Generate synthetic right camera image from left frame.
        
        This simulates what a second camera would see, positioned
        'baseline_mm' to the right of the first camera.
        
        The key insight is:
        - In a real stereo setup, objects appear at different horizontal
          positions in left vs right cameras
        - The difference (disparity) is proportional to inverse depth:
          disparity = (focal_length * baseline) / depth
        - Closer objects have LARGER disparity
        - Farther objects have SMALLER disparity
        
        Args:
            left_frame: The original webcam frame (left camera view)
            
        Returns:
            Synthetic right camera view
        """
        # Get actual frame dimensions
        h, w = left_frame.shape[:2]
        
        # Calculate disparity based on simulated depth
        # disparity = (f * B) / Z
        # Where:
        #   f = focal length in pixels
        #   B = baseline in mm
        #   Z = depth in mm
        disparity_pixels = (
            self._config.focal_length_px * self._config.baseline_mm
        ) / self._simulated_depth_mm
        
        # Create the transformation matrix for horizontal shift
        # This is a 2x3 affine transformation matrix:
        # [1, 0, tx]  where tx = -disparity (shift left to simulate right camera)
        # [0, 1, 0]
        #
        # We shift LEFT because: in a stereo pair, the right camera sees
        # objects shifted to the LEFT compared to the left camera
        shift_matrix = np.float32([
            [1, 0, -disparity_pixels],  # Shift left by disparity
            [0, 1, 0]                    # No vertical shift
        ])
        
        # Apply the transformation
        # cv2.warpAffine applies an affine transformation to an image
        # - left_frame: input image
        # - shift_matrix: 2x3 transformation matrix
        # - (w, h): output image size (use actual frame size)
        # - borderMode: how to handle pixels that go off-screen (replicate edge)
        right_frame = cv2.warpAffine(
            left_frame,
            shift_matrix,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE  # Fill gaps by repeating edge pixels
        )
        
        # Optionally add noise to simulate camera sensor differences
        if self._add_noise:
            right_frame = self._add_sensor_noise(right_frame)
        
        return right_frame
    
    def _add_sensor_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Add realistic camera sensor noise to a frame.
        
        Real cameras have slightly different sensor characteristics,
        which causes minor differences even in identical lighting.
        
        This adds Gaussian noise to simulate that effect, making the
        mock data more realistic for testing stereo algorithms.
        
        Args:
            frame: Input image (uint8 BGR)
            
        Returns:
            Image with added noise
        """
        # Generate Gaussian noise
        # - Mean = 0 (no brightness change on average)
        # - Std = 3 (small amount of variation)
        noise = np.random.normal(0, 3, frame.shape).astype(np.float32)
        
        # Add noise to frame (convert to float, add, convert back)
        noisy_frame = frame.astype(np.float32) + noise
        
        # Clip to valid range [0, 255] and convert back to uint8
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        
        return noisy_frame
    
    def set_simulated_depth(self, depth_mm: float) -> None:
        """
        Change the simulated scene depth.
        
        This affects how much disparity is in the synthetic right image.
        - Smaller depth = larger disparity (closer objects)
        - Larger depth = smaller disparity (farther objects)
        
        Use this to test your depth calculation at different ranges.
        
        Args:
            depth_mm: New simulated depth in millimeters
        """
        if depth_mm <= 0:
            raise ValueError("Depth must be positive")
        self._simulated_depth_mm = depth_mm
    
    @property
    def simulated_depth(self) -> float:
        """Get the current simulated depth in mm."""
        return self._simulated_depth_mm
    
    def release(self) -> None:
        """
        Release the webcam.
        
        Always call this when done to free the camera for other applications.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self._cap is not None and self._cap.isOpened()
    
    def get_camera_info(self) -> dict:
        """
        Get information about the camera.
        
        Returns:
            Dictionary with camera properties
        """
        if self._cap is None:
            return {}
        
        return {
            'resolution': self._actual_resolution,
            'fps': self._actual_fps,
            'camera_index': self._camera_index,
            'config_resolution': self._config.resolution,
            'config_focal_length': self._config.focal_length_px,
            'baseline_mm': self._config.baseline_mm,
        }
