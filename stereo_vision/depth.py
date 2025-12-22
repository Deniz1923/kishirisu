"""
Stereo Depth Calculator
========================

This module computes depth from stereo image pairs using OpenCV's
Semi-Global Block Matching (SGBM) algorithm.

Theory of Stereo Depth:
----------------------
When two cameras view the same scene from slightly different positions,
the same object appears at different horizontal positions in left vs right
images. This difference is called "disparity".

The relationship between disparity and depth is:

    depth = (focal_length × baseline) / disparity

Where:
- focal_length: camera focal length in pixels
- baseline: distance between cameras in mm  
- disparity: horizontal pixel difference between left and right views

This module handles:
1. Stereo rectification (aligning images for disparity computation)
2. Computing disparity map using SGBM algorithm
3. Converting disparity to depth in millimeters
4. Converting 2D pixel + depth to 3D world coordinates
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from .config import StereoConfig


class DepthCalculator:
    """
    Computes depth from stereo image pairs.
    
    This class uses OpenCV's Semi-Global Block Matching (SGBM) algorithm
    to compute disparity maps, then converts disparity to depth.
    
    SGBM is chosen over simple Block Matching (BM) because:
    - Better results in areas with low texture
    - More accurate at object boundaries
    - Handles lighting differences better
    - Worth the extra computation cost for robotics applications
    
    Example:
        >>> config = StereoConfig(baseline_mm=60.0)
        >>> depth_calc = DepthCalculator(config)
        >>> 
        >>> # Capture stereo pair from camera
        >>> left, right = camera.capture()
        >>> 
        >>> # Compute depth map
        >>> depth_map = depth_calc.compute(left, right)
        >>> 
        >>> # Get depth at a specific pixel (e.g., where a ball was detected)
        >>> depth_mm = depth_calc.get_depth_at(depth_map, x=320, y=240)
        >>> print(f"Object is {depth_mm:.0f}mm away")
        >>> 
        >>> # Convert to 3D coordinates
        >>> x_3d, y_3d, z_3d = depth_calc.pixel_to_3d(320, 240, depth_mm)
    """
    
    def __init__(self, config: StereoConfig):
        """
        Initialize the depth calculator.
        
        Args:
            config: StereoConfig with camera parameters.
                   Important parameters:
                   - baseline_mm: Distance between cameras
                   - focal_length_px: Camera focal length in pixels
                   - num_disparities: Disparity search range
                   - block_size: Matching window size
        """
        self._config = config
        
        # Create the SGBM stereo matcher
        # See _create_stereo_matcher for detailed parameter explanation
        self._stereo_matcher = self._create_stereo_matcher()
        
        # Pre-compute the disparity-to-depth conversion factor
        # depth = (focal_length * baseline) / disparity
        # So: depth = factor / disparity
        # where factor = focal_length * baseline
        self._depth_factor = config.focal_length_px * config.baseline_mm
    
    def _create_stereo_matcher(self) -> cv2.StereoSGBM:
        """
        Create and configure the SGBM stereo matcher.
        
        SGBM (Semi-Global Block Matching) works by:
        1. For each pixel in the left image, search for matching pixel in right
        2. Use a window (block) of pixels for matching, not just single pixels
        3. Apply smoothness constraints across the image (semi-global optimization)
        4. Output disparity = how many pixels the match is shifted
        
        Returns:
            Configured cv2.StereoSGBM matcher
        """
        # Minimum disparity - usually 0 unless cameras are cross-eyed
        min_disparity = 0
        
        # Number of disparities to search (must be divisible by 16)
        # Larger = can detect closer objects, but slower
        num_disparities = self._config.num_disparities
        
        # Block size for matching (must be odd, ≥5)
        # Larger = smoother but less detailed depth map
        block_size = self._config.block_size
        
        # ---- SGBM-specific parameters ----
        # These control the smoothness constraint strength
        
        # P1: Penalty for disparity changes of ±1 between neighbor pixels
        # Small P1 = allow more disparity variation (more detail, more noise)
        P1 = 8 * 3 * block_size**2  # 3 = number of color channels
        
        # P2: Penalty for disparity changes of >1 between neighbor pixels
        # Must be > P1. Large P2 = enforce smooth surfaces
        P2 = 32 * 3 * block_size**2
        
        # Create the stereo matcher with all parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=P1,                    # Smoothness penalty (small changes)
            P2=P2,                    # Smoothness penalty (large changes)
            disp12MaxDiff=1,          # Max allowed difference in L-R consistency check
            uniquenessRatio=15,       # Margin for best match to be considered unique
            speckleWindowSize=100,    # Max size of smooth disparity regions to filter
            speckleRange=32,          # Max disparity variation within speckle region
            preFilterCap=63,          # Truncation value for prefiltered image pixels
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Full SGBM for best quality
        )
        
        return stereo
    
    def compute(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        filter_invalid: bool = True
    ) -> np.ndarray:
        """
        Compute depth map from stereo image pair.
        
        This is the main method for depth calculation. It:
        1. Converts images to grayscale (SGBM works on grayscale)
        2. Computes disparity map using SGBM
        3. Filters invalid disparities
        4. Converts disparity to depth in millimeters
        
        Args:
            left_frame: Left camera image (BGR, HxWx3)
            right_frame: Right camera image (BGR, HxWx3)
            filter_invalid: If True, invalid depths are set to 0
            
        Returns:
            Depth map as float32 array (HxW) in millimeters.
            Invalid/unknown depths are set to 0 if filter_invalid=True,
            otherwise they may be inf or negative.
            
        Performance Note:
            On a typical laptop, this takes ~20-50ms per frame.
            For real-time robotics, you may need to:
            - Reduce resolution
            - Use ROI (region of interest)
            - Use GPU acceleration
        """
        # Step 1: Convert to grayscale
        # SGBM algorithm works on grayscale images
        # (color information doesn't help with stereo matching)
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Compute disparity map
        # The matcher returns disparity in fixed-point format (scaled by 16)
        # So we divide by 16 to get actual pixel disparity
        disparity_fixed = self._stereo_matcher.compute(left_gray, right_gray)
        disparity = disparity_fixed.astype(np.float32) / 16.0
        
        # Step 3: Convert disparity to depth
        # depth = (focal_length * baseline) / disparity
        # We pre-computed depth_factor = focal_length * baseline
        # 
        # Note: Where disparity <= 0, depth will be inf or negative
        # These are invalid measurements (no match found)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth_map = self._depth_factor / disparity
        
        # Step 4: Filter invalid values
        if filter_invalid:
            # Set invalid depths to 0
            # Invalid = negative, inf, NaN, or outside valid range
            invalid_mask = (
                (disparity <= 0) |
                (np.isinf(depth_map)) |
                (np.isnan(depth_map)) |
                (depth_map < self._config.min_depth_mm) |
                (depth_map > self._config.max_depth_mm)
            )
            depth_map[invalid_mask] = 0
        
        return depth_map
    
    def get_depth_at(
        self,
        depth_map: np.ndarray,
        x: int,
        y: int,
        window_size: int = 5
    ) -> float:
        """
        Get depth at a specific pixel location.
        
        Since individual depth measurements can be noisy, this method
        takes the median depth from a small window around the target pixel.
        
        Args:
            depth_map: Depth map from compute()
            x: X coordinate (horizontal, 0 = left edge)
            y: Y coordinate (vertical, 0 = top edge)
            window_size: Size of window for median filtering (odd number)
            
        Returns:
            Depth in millimeters at the given location.
            Returns 0 if depth is invalid or out of bounds.
            
        Example:
            >>> depth = depth_calc.get_depth_at(depth_map, 320, 240)
            >>> if depth > 0:
            ...     print(f"Object at {depth:.0f}mm")
            ... else:
            ...     print("No valid depth at this location")
        """
        # Validate bounds
        h, w = depth_map.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0
        
        # Calculate window bounds (ensuring we stay within image)
        half = window_size // 2
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)
        
        # Extract window and get valid (non-zero) depths
        window = depth_map[y1:y2, x1:x2]
        valid_depths = window[window > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Return median of valid depths (robust to outliers)
        return float(np.median(valid_depths))
    
    def pixel_to_3d(
        self,
        x: int,
        y: int,
        depth_mm: float
    ) -> Tuple[float, float, float]:
        """
        Convert 2D pixel coordinates + depth to 3D world coordinates.
        
        This uses the pinhole camera model to project from image space
        to 3D camera-centered coordinates.
        
        The camera coordinate system is:
        - X: right (positive values to the right of camera)
        - Y: down (positive values below camera center)
        - Z: forward (depth, into the scene)
        
        The math:
            X_3d = (x - cx) * depth / focal_length
            Y_3d = (y - cy) * depth / focal_length
            Z_3d = depth
            
        Where (cx, cy) is the principal point (optical center).
        
        Args:
            x: Pixel X coordinate
            y: Pixel Y coordinate
            depth_mm: Depth at that pixel in millimeters
            
        Returns:
            Tuple of (X, Y, Z) in millimeters in camera coordinate frame.
            
        Example:
            >>> # Ball detected at pixel (400, 300), depth 800mm
            >>> x_3d, y_3d, z_3d = depth_calc.pixel_to_3d(400, 300, 800)
            >>> print(f"Ball is at ({x_3d:.0f}, {y_3d:.0f}, {z_3d:.0f}) mm")
        """
        if depth_mm <= 0:
            return (0.0, 0.0, 0.0)
        
        # Get camera intrinsics
        cx = self._config.cx
        cy = self._config.cy
        f = self._config.focal_length_px
        
        # Project to 3D using pinhole camera model
        x_3d = (x - cx) * depth_mm / f
        y_3d = (y - cy) * depth_mm / f
        z_3d = depth_mm
        
        return (x_3d, y_3d, z_3d)
    
    def compute_disparity_map(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray
    ) -> np.ndarray:
        """
        Compute raw disparity map (for visualization/debugging).
        
        Unlike compute(), this returns disparity values (in pixels)
        rather than depth (in mm). Useful for visualizing or debugging
        the stereo matching quality.
        
        Args:
            left_frame: Left camera image (BGR)
            right_frame: Right camera image (BGR)
            
        Returns:
            Disparity map as float32 (HxW), values in pixels.
            Negative values indicate invalid matches.
        """
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        disparity_fixed = self._stereo_matcher.compute(left_gray, right_gray)
        return disparity_fixed.astype(np.float32) / 16.0
    
    def visualize_depth(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None
    ) -> np.ndarray:
        """
        Create a colorized visualization of the depth map.
        
        Converts depth values to a color image for display:
        - Closer objects (small depth) = warm colors (red/orange)
        - Farther objects (large depth) = cool colors (blue/purple)
        - Invalid depths (0) = black
        
        Args:
            depth_map: Depth map from compute()
            colormap: OpenCV colormap (cv2.COLORMAP_JET, etc.)
            min_depth: Minimum depth for color scale (None = auto)
            max_depth: Maximum depth for color scale (None = auto)
            
        Returns:
            Colorized depth visualization as BGR image (HxWx3)
        """
        # Get depth range
        if min_depth is None:
            min_depth = self._config.min_depth_mm
        if max_depth is None:
            max_depth = self._config.max_depth_mm
        
        # Create a mask of valid depths
        valid_mask = depth_map > 0
        
        # Normalize depth to 0-255 range (inverted so close = high value)
        # Close objects get high values (will be warm colors with JET)
        normalized = np.zeros_like(depth_map, dtype=np.uint8)
        if np.any(valid_mask):
            # Invert and normalize
            depth_normalized = (max_depth - np.clip(depth_map, min_depth, max_depth)) / (max_depth - min_depth)
            normalized[valid_mask] = (depth_normalized[valid_mask] * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)
        
        # Set invalid regions to black
        colored[~valid_mask] = [0, 0, 0]
        
        return colored
    
    @property
    def config(self) -> StereoConfig:
        """Get the stereo configuration."""
        return self._config
