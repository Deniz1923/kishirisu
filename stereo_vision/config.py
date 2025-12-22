"""
Stereo Camera Configuration
============================

This module contains configuration dataclasses for the stereo camera system.
All measurements are in millimeters (mm) unless otherwise specified.

The configuration includes:
- Camera intrinsic parameters (focal length, principal point)
- Stereo baseline (distance between cameras)
- Camera resolution (can be auto-detected)
- X/Y axis movement ranges for the movable camera mount
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import cv2


def detect_camera_capabilities(camera_index: int = 0) -> dict:
    """
    Detect camera resolution and FPS capabilities.
    
    Opens the camera briefly to query its actual capabilities.
    This is useful for auto-configuring the stereo system.
    
    Args:
        camera_index: OpenCV camera index (0 = default webcam)
        
    Returns:
        Dictionary with detected capabilities:
        {
            'resolution': (width, height),
            'fps': frames_per_second,
            'backend': camera backend name,
            'name': camera name if available
        }
        
    Raises:
        RuntimeError: If camera cannot be opened
        
    Example:
        >>> caps = detect_camera_capabilities(0)
        >>> print(f"Camera: {caps['resolution'][0]}x{caps['resolution'][1]} @ {caps['fps']} FPS")
        Camera: 1920x1080 @ 30.0 FPS
    """
    # Open camera connection
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera at index {camera_index}. "
            "Make sure your webcam is connected and not in use."
        )
    
    try:
        # Try to set maximum resolution (camera will use highest it supports)
        # Common HD resolutions to try
        resolutions_to_try = [
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (640, 480),    # VGA
        ]
        
        best_resolution = (640, 480)
        
        for width, height in resolutions_to_try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Check what we actually got
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_w >= best_resolution[0]:
                best_resolution = (actual_w, actual_h)
                break
        
        # Get FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default assumption
        
        # Get backend info
        backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))
        backend_name = "Unknown"
        try:
            backend_name = cap.getBackendName()
        except:
            pass
        
        return {
            'resolution': best_resolution,
            'fps': fps,
            'backend': backend_name,
            'camera_index': camera_index,
        }
        
    finally:
        cap.release()


@dataclass
class StereoConfig:
    """
    Configuration for stereo camera system.
    
    This class holds all parameters needed for stereo depth calculation.
    Set resolution to None for auto-detection from camera hardware.
    
    Attributes:
        resolution: (width, height) in pixels, or None for auto-detect
        fps: Target FPS, or None for auto-detect
        focal_length_px: Camera focal length in pixels. Set to None to estimate
                        from resolution (assumes ~70° horizontal FOV typical webcam)
        principal_point: (cx, cy) optical center, None to auto-calculate as image center
        baseline_mm: Distance between left and right cameras in millimeters
        x_axis_range: (min, max) movement range on X-axis in mm
        y_axis_range: (min, max) movement range on Y-axis in mm
        min_depth_mm: Minimum reliable depth measurement
        max_depth_mm: Maximum reliable depth measurement
        
    Example:
        >>> # Auto-detect from camera
        >>> config = StereoConfig.from_camera(camera_index=0)
        >>> print(f"Detected: {config.resolution} @ {config.fps} FPS")
        
        >>> # Manual configuration
        >>> config = StereoConfig(resolution=(1920, 1080), fps=60)
    """
    
    # ----- Camera Resolution -----
    # Width and height of camera frames in pixels
    # Set to None for auto-detection
    resolution: Tuple[int, int] = (640, 480)
    
    # ----- Frame Rate -----
    # Target frames per second, None for auto-detect
    fps: float = 30.0
    
    # ----- Camera Intrinsics -----
    # Focal length in pixels
    # Set to None to auto-estimate based on resolution (assumes ~70° HFOV)
    # Formula: focal_length_px = width / (2 * tan(HFOV/2))
    # For 70° HFOV: focal_length_px ≈ width * 0.82
    focal_length_px: Optional[float] = None
    
    # Principal point (optical center), typically at image center
    # Set to None to auto-calculate as (width/2, height/2)
    principal_point: Optional[Tuple[float, float]] = None
    
    # ----- Stereo Configuration -----
    # Distance between left and right camera centers in millimeters
    # Larger baseline = better depth precision at long range, worse at close range
    baseline_mm: float = 60.0  # Typical webcam stereo baseline
    
    # ----- Axis Movement Ranges -----
    # The cameras on your robot can move on X (horizontal) and Y (vertical) axes
    # These define the movement limits in millimeters from center position
    x_axis_range: Tuple[float, float] = (-50.0, 50.0)  # Left/Right movement
    y_axis_range: Tuple[float, float] = (-30.0, 30.0)  # Up/Down movement
    
    # ----- Depth Range -----
    # Valid depth measurement range in millimeters
    # Objects closer than min or farther than max will have unreliable depth
    min_depth_mm: float = 200.0    # 20cm minimum
    max_depth_mm: float = 5000.0   # 5m maximum
    
    # ----- Stereo Matching Parameters -----
    # These affect the quality and speed of depth computation
    num_disparities: int = 64      # Must be divisible by 16, higher = larger depth range
    block_size: int = 11           # Odd number, larger = smoother but less detail
    
    # ----- Auto-detection metadata (populated after from_camera()) -----
    _detected_fps: Optional[float] = field(default=None, repr=False)
    _camera_backend: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """
        Post-initialization validation and auto-calculation of derived values.
        """
        # Validate resolution
        assert self.resolution is not None, \
            "Resolution must be set (use from_camera() for auto-detection)"
        assert self.resolution[0] > 0 and self.resolution[1] > 0, \
            "Resolution must be positive"
        
        # Auto-estimate focal length if not provided
        # Assumes typical webcam with ~70° horizontal field of view
        # focal_length = width / (2 * tan(35°)) ≈ width * 0.82
        if self.focal_length_px is None:
            self.focal_length_px = self.resolution[0] * 0.82
        
        # Auto-calculate principal point if not provided
        if self.principal_point is None:
            self.principal_point = (
                self.resolution[0] / 2.0,
                self.resolution[1] / 2.0
            )
        
        # Validate other parameters
        assert self.focal_length_px > 0, \
            "Focal length must be positive"
        assert self.baseline_mm > 0, \
            "Baseline must be positive"
        assert self.num_disparities % 16 == 0, \
            "num_disparities must be divisible by 16"
        assert self.block_size % 2 == 1 and self.block_size >= 5, \
            "block_size must be odd and >= 5"
    
    @classmethod
    def from_camera(
        cls,
        camera_index: int = 0,
        baseline_mm: float = 60.0,
        prefer_resolution: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> "StereoConfig":
        """
        Create configuration by auto-detecting camera capabilities.
        
        This is the recommended way to create a StereoConfig when you want
        to use the camera's native resolution and FPS.
        
        Args:
            camera_index: OpenCV camera index (0 = default webcam)
            baseline_mm: Stereo baseline in mm (still needs manual measurement)
            prefer_resolution: If set, try to use this resolution instead of max
            **kwargs: Additional StereoConfig parameters to override
            
        Returns:
            StereoConfig with detected resolution and FPS
            
        Example:
            >>> # Auto-detect everything
            >>> config = StereoConfig.from_camera(camera_index=0)
            >>> print(f"Using: {config.resolution} @ {config.fps} FPS")
            
            >>> # Auto-detect but prefer 720p
            >>> config = StereoConfig.from_camera(prefer_resolution=(1280, 720))
        """
        print(f"[StereoConfig] Detecting camera {camera_index} capabilities...")
        
        caps = detect_camera_capabilities(camera_index)
        
        resolution = caps['resolution']
        fps = caps['fps']
        
        print(f"[StereoConfig] Detected: {resolution[0]}x{resolution[1]} @ {fps} FPS")
        print(f"[StereoConfig] Backend: {caps['backend']}")
        
        # If preferred resolution requested, try to use it
        if prefer_resolution is not None:
            # Validate it's <= detected max
            if (prefer_resolution[0] <= resolution[0] and 
                prefer_resolution[1] <= resolution[1]):
                resolution = prefer_resolution
                print(f"[StereoConfig] Using preferred: {resolution[0]}x{resolution[1]}")
        
        config = cls(
            resolution=resolution,
            fps=fps,
            baseline_mm=baseline_mm,
            _detected_fps=fps,
            _camera_backend=caps['backend'],
            **kwargs
        )
        
        return config
    
    @property
    def width(self) -> int:
        """Get image width in pixels."""
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        """Get image height in pixels."""
        return self.resolution[1]
    
    @property
    def cx(self) -> float:
        """Get principal point X coordinate."""
        return self.principal_point[0]
    
    @property
    def cy(self) -> float:
        """Get principal point Y coordinate."""
        return self.principal_point[1]
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height)."""
        return self.resolution[0] / self.resolution[1]
    
    def get_camera_matrix(self):
        """
        Get the 3x3 camera intrinsic matrix K.
        
        The camera matrix is:
            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        
        Where fx=fy=focal_length_px (assuming square pixels)
        
        Returns:
            numpy.ndarray: 3x3 camera intrinsic matrix
        """
        import numpy as np
        return np.array([
            [self.focal_length_px, 0, self.cx],
            [0, self.focal_length_px, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def scale_to_resolution(self, new_resolution: Tuple[int, int]) -> "StereoConfig":
        """
        Create a new config scaled to a different resolution.
        
        Useful for processing at lower resolution for speed,
        then mapping results back to full resolution.
        
        Args:
            new_resolution: Target (width, height)
            
        Returns:
            New StereoConfig scaled appropriately
        """
        scale_x = new_resolution[0] / self.resolution[0]
        scale_y = new_resolution[1] / self.resolution[1]
        scale = (scale_x + scale_y) / 2  # Average scale
        
        return StereoConfig(
            resolution=new_resolution,
            fps=self.fps,
            focal_length_px=self.focal_length_px * scale,
            principal_point=(self.cx * scale_x, self.cy * scale_y),
            baseline_mm=self.baseline_mm,
            x_axis_range=self.x_axis_range,
            y_axis_range=self.y_axis_range,
            min_depth_mm=self.min_depth_mm,
            max_depth_mm=self.max_depth_mm,
            num_disparities=self.num_disparities,
            block_size=self.block_size,
        )
