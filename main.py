#!/usr/bin/env python3
"""
Ball Catcher Robot - Stereo Vision Demo
=========================================

This demo application shows how all the stereo vision modules work together:
1. Capture stereo pairs (mock mode using single webcam)
2. Calculate depth map from stereo images
3. Detect objects in the frame
4. Calculate 3D position of detected objects

Controls:
---------
- Arrow keys: Move simulated camera position (X/Y axes)
- +/-: Increase/decrease simulated depth
- Space: Reset camera to center
- 'c': Toggle colormap for depth visualization
- 'd': Toggle depth map overlay
- 's': Save current frame
- 'q' or ESC: Quit

Usage:
------
    # Run the demo
    uv run python main.py
    
    # Run with custom parameters
    uv run python main.py --baseline 100 --depth 500

Requirements:
------------
- Webcam connected to your computer
- OpenCV and NumPy installed (via uv sync)

Author: Ball Catcher Robot Team
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import our stereo vision modules
from stereo_vision import (
    StereoConfig,
    MockStereoCamera,
    DepthCalculator,
    DummyDetector,
    Detection,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Stereo Vision Depth Calculation Demo for Ball Catcher Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py                    # Run with defaults
  uv run python main.py --baseline 100     # 100mm camera baseline
  uv run python main.py --depth 500        # Start with 500mm simulated depth
  uv run python main.py --no-detector      # Disable object detection
        """
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index (default: 0 = first webcam)"
    )
    parser.add_argument(
        "--baseline", "-b",
        type=float,
        default=60.0,
        help="Stereo baseline in mm (default: 60)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Display scale factor (default: 0.5 to fit screen)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=float,
        default=1000.0,
        help="Initial simulated depth in mm (default: 1000)"
    )
    parser.add_argument(
        "--focal-length", "-f",
        type=float,
        default=None,
        help="Camera focal length in pixels (default: auto-estimate from resolution)"
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        default=True,
        help="Auto-detect camera resolution and FPS (default: True)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        default="640x480",
        help="Preferred resolution WxH (default: 640x480 for performance)"
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="Disable object detection (runs faster)"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("captures"),
        help="Directory to save captured frames"
    )
    
    return parser.parse_args()


def create_info_panel(
    width: int,
    camera_x: float,
    camera_y: float,
    simulated_depth: float,
    fps: float,
    detections: list
) -> np.ndarray:
    """
    Create an information panel showing current state.
    
    Args:
        width: Panel width in pixels
        camera_x: Current camera X position
        camera_y: Current camera Y position
        simulated_depth: Current simulated depth
        fps: Current frames per second
        detections: List of current detections
        
    Returns:
        BGR image of the info panel
    """
    # Create dark panel
    height = 120 + len(detections) * 25
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    # Drawing settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (200, 200, 200)  # Light gray text
    y_offset = 25
    
    # Title
    cv2.putText(panel, "=== STEREO VISION STATUS ===", (10, y_offset), 
                font, 0.6, (100, 255, 100), 1)
    y_offset += 30
    
    # Camera position
    cv2.putText(panel, f"Camera Position: X={camera_x:+.1f}mm  Y={camera_y:+.1f}mm", 
                (10, y_offset), font, font_scale, color, 1)
    y_offset += 25
    
    # Simulated depth
    cv2.putText(panel, f"Simulated Depth: {simulated_depth:.0f}mm", 
                (10, y_offset), font, font_scale, color, 1)
    y_offset += 25
    
    # FPS
    fps_color = (100, 255, 100) if fps >= 15 else (100, 100, 255)
    cv2.putText(panel, f"FPS: {fps:.1f}", 
                (10, y_offset), font, font_scale, fps_color, 1)
    y_offset += 30
    
    # Detections
    if detections:
        cv2.putText(panel, f"Detections ({len(detections)}):", 
                    (10, y_offset), font, font_scale, (255, 200, 100), 1)
        y_offset += 25
        
        for det in detections:
            text = f"  {det.label}: ({det.x}, {det.y}) conf={det.confidence:.2f}"
            cv2.putText(panel, text, (10, y_offset), font, 0.4, color, 1)
            y_offset += 20
    else:
        cv2.putText(panel, "No detections", 
                    (10, y_offset), font, font_scale, (150, 150, 150), 1)
    
    return panel


def create_display_frame(
    left: np.ndarray,
    right: np.ndarray,
    depth_colored: np.ndarray,
    info_panel: np.ndarray,
    show_depth_overlay: bool = True,
    scale: float = 1.0
) -> np.ndarray:
    """
    Combine all visualization elements into a single display frame.
    
    Layout:
    +----------------+----------------+
    |  LEFT CAMERA   |  RIGHT CAMERA  |
    +----------------+----------------+
    |   DEPTH MAP    |   INFO PANEL   |
    +----------------+----------------+
    
    Args:
        left: Left camera frame
        right: Right camera frame
        depth_colored: Colorized depth map
        info_panel: Information panel
        show_depth_overlay: Whether to blend depth with left image
        
    Returns:
        Combined display frame
    """
    h, w = left.shape[:2]
    
    # Add labels to camera frames
    left_labeled = left.copy()
    right_labeled = right.copy()
    cv2.putText(left_labeled, "LEFT", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(right_labeled, "RIGHT", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Create top row (left + right cameras)
    top_row = np.hstack([left_labeled, right_labeled])
    
    # Create depth display
    if show_depth_overlay:
        # Blend depth with left camera image
        alpha = 0.6
        depth_blend = cv2.addWeighted(left, 1 - alpha, depth_colored, alpha, 0)
        cv2.putText(depth_blend, "DEPTH", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        depth_blend = depth_colored.copy()
        cv2.putText(depth_blend, "DEPTH", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Resize info panel to match depth map height
    info_resized = cv2.resize(info_panel, (w, h))
    
    # Create bottom row (depth + info)
    bottom_row = np.hstack([depth_blend, info_resized])
    
    # Combine top and bottom
    display = np.vstack([top_row, bottom_row])
    
    # Scale down for display if needed
    if scale != 1.0:
        new_w = int(display.shape[1] * scale)
        new_h = int(display.shape[0] * scale)
        display = cv2.resize(display, (new_w, new_h))
    
    return display


def main():
    """
    Main demo loop.
    
    This function:
    1. Initializes all stereo vision components
    2. Runs the main capture/process/display loop
    3. Handles keyboard input for interaction
    """
    print("=" * 60)
    print("  STEREO VISION DEPTH DEMO - Ball Catcher Robot")
    print("=" * 60)
    print()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # ========================================
    # STEP 1: Configure the stereo system
    # ========================================
    print("[1/4] Configuring stereo system...")
    
    # Parse preferred resolution if specified
    prefer_resolution = None
    if args.resolution:
        try:
            w, h = args.resolution.lower().split('x')
            prefer_resolution = (int(w), int(h))
            print(f"    Preferred resolution: {prefer_resolution[0]}x{prefer_resolution[1]}")
        except ValueError:
            print(f"    Warning: Invalid resolution '{args.resolution}', using auto-detect")
    
    # Auto-detect camera capabilities or use manual config
    if args.auto:
        # Auto-detect from camera hardware
        config = StereoConfig.from_camera(
            camera_index=args.camera,
            baseline_mm=args.baseline,
            prefer_resolution=prefer_resolution,
        )
        # Override focal length if manually specified
        if args.focal_length is not None:
            config.focal_length_px = args.focal_length
    else:
        # Manual configuration
        resolution = prefer_resolution or (640, 480)
        config = StereoConfig(
            resolution=resolution,
            focal_length_px=args.focal_length or (resolution[0] * 0.82),
            baseline_mm=args.baseline,
            x_axis_range=(-50.0, 50.0),
            y_axis_range=(-30.0, 30.0),
        )
    
    print(f"    Baseline: {config.baseline_mm}mm")
    print(f"    Focal length: {config.focal_length_px:.1f}px (estimated from ~70° HFOV)")
    print(f"    Resolution: {config.resolution[0]}x{config.resolution[1]}")
    print(f"    Target FPS: {config.fps}")
    
    # ========================================
    # STEP 2: Initialize components
    # ========================================
    print("[2/4] Initializing camera...")
    
    try:
        # Initialize mock stereo camera using the laptop webcam
        # In a real deployment, this would be replaced with RealStereoCamera
        camera = MockStereoCamera(
            config=config,
            camera_index=args.camera,
            simulated_depth_mm=args.depth,
            add_noise=True  # Add realistic noise for testing
        )
        print(f"    Camera {args.camera} opened successfully")
    except RuntimeError as e:
        print(f"    ERROR: {e}")
        print("    Make sure your webcam is connected and not in use by another app.")
        sys.exit(1)
    
    print("[3/4] Initializing depth calculator...")
    
    # Initialize depth calculator with our configuration
    depth_calc = DepthCalculator(config)
    print(f"    Num disparities: {config.num_disparities}")
    print(f"    Block size: {config.block_size}")
    
    print("[4/4] Initializing detector...")
    
    # Initialize object detector (dummy/placeholder until YOLO is ready)
    if not args.no_detector:
        # This is a simple color-based detector for testing
        # Your friend will replace this with YOLO
        detector = DummyDetector(
            label="ball",
            # HSV range for orange/red objects
            hsv_lower=(0, 100, 100),
            hsv_upper=(20, 255, 255),
            min_area=500
        )
        print("    Using DummyDetector (color-based)")
        print("    TIP: Your friend should replace this with YOLODetector")
    else:
        detector = None
        print("    Detection disabled")
    
    # Create save directory if needed
    if not args.save_dir.exists():
        args.save_dir.mkdir(parents=True)
    
    # ========================================
    # STEP 3: Main loop
    # ========================================
    print()
    print("=" * 60)
    print("  CONTROLS:")
    print("    Arrow keys  - Move camera X/Y")
    print("    +/-         - Adjust simulated depth")
    print("    Space       - Reset camera position")
    print("    D           - Toggle depth overlay")
    print("    S           - Save current frame")
    print("    Q/ESC       - Quit")
    print("=" * 60)
    print()
    print("Starting capture loop... Press 'q' to quit.")
    
    # State variables
    show_depth_overlay = True
    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()
    detections = []
    
    # Colormap options for depth visualization
    colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_RAINBOW]
    colormap_idx = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # ---- Capture stereo pair ----
            try:
                left, right = camera.capture()
            except RuntimeError as e:
                print(f"Capture error: {e}")
                break
            
            # ---- Compute depth ----
            depth_map = depth_calc.compute(left, right)
            
            # ---- Colorize depth for visualization ----
            depth_colored = depth_calc.visualize_depth(
                depth_map,
                colormap=colormaps[colormap_idx]
            )
            
            # ---- Detect objects ----
            if detector is not None:
                detections = detector.detect(left)
                
                # Draw detections on the left frame
                for det in detections:
                    det.draw_on(left, color=(0, 255, 0))
                    
                    # Get depth at detection center
                    depth_at_det = depth_calc.get_depth_at(depth_map, det.x, det.y)
                    
                    if depth_at_det > 0:
                        # Calculate 3D position
                        x_3d, y_3d, z_3d = depth_calc.pixel_to_3d(
                            det.x, det.y, depth_at_det
                        )
                        
                        # Draw 3D position on depth map
                        pos_text = f"3D: ({x_3d:.0f}, {y_3d:.0f}, {z_3d:.0f})mm"
                        cv2.putText(
                            depth_colored, pos_text, (det.x - 50, det.y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                        )
            
            # ---- Create info panel ----
            info_panel = create_info_panel(
                width=config.width,
                camera_x=camera.current_x,
                camera_y=camera.current_y,
                simulated_depth=camera.simulated_depth,
                fps=fps,
                detections=detections
            )
            
            # ---- Create combined display ----
            display = create_display_frame(
                left, right, depth_colored, info_panel,
                show_depth_overlay=show_depth_overlay,
                scale=args.scale
            )
            
            # ---- Show the display ----
            cv2.imshow("Stereo Vision Demo", display)
            
            # ---- Handle keyboard input ----
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("Quitting...")
                break
            
            elif key == ord('d'):  # Toggle depth overlay
                show_depth_overlay = not show_depth_overlay
                print(f"Depth overlay: {'ON' if show_depth_overlay else 'OFF'}")
            
            elif key == ord('c'):  # Cycle colormap
                colormap_idx = (colormap_idx + 1) % len(colormaps)
                names = ["JET", "HOT", "RAINBOW"]
                print(f"Colormap: {names[colormap_idx]}")
            
            elif key == ord('s'):  # Save frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = args.save_dir / f"capture_{timestamp}.png"
                cv2.imwrite(str(save_path), display)
                print(f"Saved: {save_path}")
            
            elif key == ord(' '):  # Reset camera position
                camera.reset_position()
                print("Camera position reset")
            
            elif key == ord('=') or key == ord('+'):  # Increase depth
                new_depth = camera.simulated_depth + 100
                camera.set_simulated_depth(new_depth)
                print(f"Simulated depth: {new_depth:.0f}mm")
            
            elif key == ord('-'):  # Decrease depth
                new_depth = max(200, camera.simulated_depth - 100)
                camera.set_simulated_depth(new_depth)
                print(f"Simulated depth: {new_depth:.0f}mm")
            
            # Arrow keys for camera movement
            elif key == 81 or key == 2:  # Left arrow
                camera.move_relative(-5, 0)
            elif key == 83 or key == 3:  # Right arrow
                camera.move_relative(5, 0)
            elif key == 82 or key == 0:  # Up arrow
                camera.move_relative(0, -5)
            elif key == 84 or key == 1:  # Down arrow
                camera.move_relative(0, 5)
            
            # ---- Update FPS calculation ----
            frame_count += 1
            elapsed = time.time() - last_fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = time.time()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # ========================================
        # STEP 4: Cleanup
        # ========================================
        print("Releasing resources...")
        camera.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
