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
- 'f': Toggle fast mode (downscaled processing)
- 'h': Toggle help overlay
- 's': Save current frame
- 'q' or ESC: Quit
- Mouse click: Print depth at clicked point

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
        "--fast", "-F",
        action="store_true",
        help="Enable fast mode (downscaled depth computation)"
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
    detections: list,
    depth_stats: dict = None,
    center_depth: float = 0,
    fast_mode: bool = False,
    process_time_ms: float = 0
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
        depth_stats: Dictionary with depth statistics
        center_depth: Depth at image center in mm
        fast_mode: Whether fast mode is enabled
        process_time_ms: Processing time in milliseconds
        
    Returns:
        BGR image of the info panel
    """
    # Create dark panel
    height = 200 + len(detections) * 25
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    # Drawing settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (200, 200, 200)  # Light gray text
    y_offset = 25
    
    # Title
    mode_text = " [FAST]" if fast_mode else ""
    cv2.putText(panel, f"=== STEREO VISION STATUS{mode_text} ===", (10, y_offset), 
                font, 0.6, (100, 255, 100), 1)
    y_offset += 30
    
    # FPS and processing time
    fps_color = (100, 255, 100) if fps >= 15 else (100, 100, 255)
    cv2.putText(panel, f"FPS: {fps:.1f}  |  Process: {process_time_ms:.1f}ms", 
                (10, y_offset), font, font_scale, fps_color, 1)
    y_offset += 25
    
    # CENTER DEPTH (prominent display)
    if center_depth > 0:
        depth_text = f"CENTER DEPTH: {center_depth:.0f}mm ({center_depth/10:.1f}cm)"
        cv2.putText(panel, depth_text, (10, y_offset), font, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(panel, "CENTER DEPTH: --", (10, y_offset), font, 0.6, (100, 100, 100), 1)
    y_offset += 30
    
    # Depth statistics
    if depth_stats:
        stats_text = f"Depth Range: {depth_stats['min_depth']:.0f} - {depth_stats['max_depth']:.0f}mm"
        cv2.putText(panel, stats_text, (10, y_offset), font, font_scale, (200, 200, 100), 1)
        y_offset += 20
        stats_text2 = f"Median: {depth_stats['median_depth']:.0f}mm  Valid: {depth_stats['valid_percent']:.1f}%"
        cv2.putText(panel, stats_text2, (10, y_offset), font, font_scale, (200, 200, 100), 1)
        y_offset += 25
    
    # Camera position
    cv2.putText(panel, f"Camera: X={camera_x:+.1f}mm  Y={camera_y:+.1f}mm", 
                (10, y_offset), font, font_scale, color, 1)
    y_offset += 20
    
    # Simulated depth
    cv2.putText(panel, f"Simulated: {simulated_depth:.0f}mm", 
                (10, y_offset), font, font_scale, color, 1)
    y_offset += 25
    
    # Detections
    if detections:
        cv2.putText(panel, f"Detections ({len(detections)}):", 
                    (10, y_offset), font, font_scale, (255, 200, 100), 1)
        y_offset += 25
        
        for det in detections[:3]:  # Limit to 3 detections to save space
            text = f"  {det.label}: ({det.x}, {det.y}) conf={det.confidence:.2f}"
            cv2.putText(panel, text, (10, y_offset), font, 0.4, color, 1)
            y_offset += 20
    else:
        cv2.putText(panel, "No detections", 
                    (10, y_offset), font, font_scale, (150, 150, 150), 1)
    
    return panel


def create_help_overlay(width: int, height: int) -> np.ndarray:
    """
    Create a semi-transparent help overlay showing keyboard shortcuts.
    """
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Semi-transparent background
    cv2.rectangle(overlay, (50, 50), (width - 50, height - 50), (40, 40, 40), -1)
    cv2.rectangle(overlay, (50, 50), (width - 50, height - 50), (100, 255, 100), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 90
    
    cv2.putText(overlay, "KEYBOARD SHORTCUTS", (70, y), font, 0.8, (100, 255, 100), 2)
    y += 40
    
    shortcuts = [
        ("Arrow Keys", "Move camera position"),
        ("+/-", "Adjust simulated depth"),
        ("Space", "Reset camera position"),
        ("C", "Cycle colormap"),
        ("D", "Toggle depth overlay"),
        ("F", "Toggle fast mode"),
        ("H", "Toggle this help"),
        ("S", "Save frame"),
        ("Q/ESC", "Quit"),
        ("Mouse Click", "Print depth at point"),
    ]
    
    for key, desc in shortcuts:
        cv2.putText(overlay, f"{key:12s}", (70, y), font, 0.5, (255, 200, 100), 1)
        cv2.putText(overlay, desc, (180, y), font, 0.5, (200, 200, 200), 1)
        y += 25
    
    return overlay


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
    print("    F           - Toggle FAST mode")
    print("    H           - Toggle help overlay")
    print("    D           - Toggle depth overlay")
    print("    S           - Save current frame")
    print("    Q/ESC       - Quit")
    print("    Click       - Print depth at mouse position")
    print("=" * 60)
    print()
    print("Starting capture loop... Press 'q' to quit.")
    print()
    
    # State variables
    show_depth_overlay = True
    show_help = False
    fast_mode = args.fast
    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()
    last_depth_print_time = time.time()
    detections = []
    depth_stats = None
    center_depth = 0.0
    process_time_ms = 0.0
    
    # Mouse callback state
    click_depth_map = None
    click_depth_calc = depth_calc
    
    def mouse_callback(event, x, y, flags, param):
        """Handle mouse clicks to print depth at clicked point."""
        nonlocal click_depth_map, click_depth_calc
        if event == cv2.EVENT_LBUTTONDOWN:
            if click_depth_map is not None:
                # Account for display scaling and layout
                # The display is: top row = left|right, bottom row = depth|info
                # Each panel is at scale
                panel_w = int(config.width * args.scale)
                panel_h = int(config.height * args.scale)
                
                # Check if click is in left camera or depth panel
                if y < panel_h:  # Top row
                    if x < panel_w:  # Left camera
                        img_x = int(x / args.scale)
                        img_y = int(y / args.scale)
                    else:
                        return  # Right camera, skip
                else:  # Bottom row
                    if x < panel_w:  # Depth panel
                        img_x = int(x / args.scale)
                        img_y = int((y - panel_h) / args.scale)
                    else:
                        return  # Info panel, skip
                
                depth_at_click = click_depth_calc.get_depth_at(click_depth_map, img_x, img_y)
                if depth_at_click > 0:
                    x_3d, y_3d, z_3d = click_depth_calc.pixel_to_3d(img_x, img_y, depth_at_click)
                    print(f"\n>>> CLICKED @ ({img_x}, {img_y}): depth={depth_at_click:.0f}mm  3D=({x_3d:.0f}, {y_3d:.0f}, {z_3d:.0f})mm <<<")
                else:
                    print(f"\n>>> CLICKED @ ({img_x}, {img_y}): No valid depth <<<")
    
    cv2.namedWindow("Stereo Vision Demo")
    cv2.setMouseCallback("Stereo Vision Demo", mouse_callback)
    
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
            
            # ---- Compute depth (with optional fast mode) ----
            depth_start = time.time()
            if fast_mode:
                depth_map = depth_calc.compute_fast(left, right, scale=0.5)
            else:
                depth_map = depth_calc.compute(left, right)
            process_time_ms = (time.time() - depth_start) * 1000
            
            # Store for mouse callback
            click_depth_map = depth_map
            
            # ---- Get depth statistics ----
            depth_stats = depth_calc.get_depth_stats(depth_map)
            center_depth = depth_calc.get_depth_at(
                depth_map, config.width // 2, config.height // 2, window_size=15
            )
            
            # ---- Colorize depth for visualization ----
            depth_colored = depth_calc.visualize_depth(
                depth_map,
                colormap=colormaps[colormap_idx]
            )
            
            # ---- Draw center crosshair and depth on depth map ----
            cx, cy = config.width // 2, config.height // 2
            cv2.drawMarker(depth_colored, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            if center_depth > 0:
                cv2.putText(depth_colored, f"{center_depth:.0f}mm", (cx + 15, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
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
                detections=detections,
                depth_stats=depth_stats,
                center_depth=center_depth,
                fast_mode=fast_mode,
                process_time_ms=process_time_ms
            )
            
            # ---- Create combined display ----
            display = create_display_frame(
                left, right, depth_colored, info_panel,
                show_depth_overlay=show_depth_overlay,
                scale=args.scale
            )
            
            # ---- Add help overlay if enabled ----
            if show_help:
                help_overlay = create_help_overlay(display.shape[1], display.shape[0])
                display = cv2.addWeighted(display, 0.7, help_overlay, 0.3, 0)
            
            # ---- Show the display ----
            cv2.imshow("Stereo Vision Demo", display)
            
            # ---- Periodic depth printing to terminal ----
            if time.time() - last_depth_print_time >= 0.5:
                last_depth_print_time = time.time()
                if center_depth > 0:
                    print(f"[DEPTH] Center: {center_depth:.0f}mm ({center_depth/10:.1f}cm) | Range: {depth_stats['min_depth']:.0f}-{depth_stats['max_depth']:.0f}mm | FPS: {fps:.1f}")
                else:
                    print(f"[DEPTH] Center: -- | FPS: {fps:.1f}")
            
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
            
            elif key == ord('f'):  # Toggle fast mode
                fast_mode = not fast_mode
                print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
            
            elif key == ord('h'):  # Toggle help overlay
                show_help = not show_help
            
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
