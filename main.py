#!/usr/bin/env python3
"""
Kishirisu - Stereo Vision Demo
===============================

Interactive demonstration of stereo depth calculation.

Controls:
    Arrow keys    Move camera position
    +/-           Adjust simulated depth
    Space         Reset camera to center
    C             Cycle colormap
    D             Toggle depth overlay
    F             Toggle fast mode
    H             Toggle help overlay
    S             Save frame
    Q/ESC         Quit
    Mouse click   Print depth at point

Usage:
    uv run python main.py
    uv run python main.py --baseline 100 --depth 500
    uv run python main.py --fast --resolution 1280x720
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from stereo_vision import (
    StereoConfig,
    MockStereoCamera,
    RealStereoCamera,
    CaptureMode,
    DepthCalculator,
    DepthStats,
    DummyDetector,
    Detection,
    Resolution,
    QualityPreset,
    StereoCapture,
)

if TYPE_CHECKING:
    import numpy.typing as npt


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DemoConfig:
    """Demo application configuration."""

    camera_index: int = 0
    baseline_mm: float = 60.0
    simulated_depth_mm: float = 1000.0
    resolution: Resolution = field(default_factory=lambda: Resolution(640, 480))
    display_scale: float = 0.5
    fast_mode: bool = False
    enable_detector: bool = True
    save_dir: Path = field(default_factory=lambda: Path("captures"))
    preset: QualityPreset = QualityPreset.BALANCED
    use_mock_camera: bool = False  # Default to real camera

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DemoConfig:
        """Create from parsed arguments."""
        resolution = Resolution.parse(args.resolution)
        return cls(
            camera_index=args.camera,
            baseline_mm=args.baseline,
            simulated_depth_mm=args.depth,
            resolution=resolution,
            display_scale=args.scale,
            fast_mode=args.fast,
            enable_detector=not args.no_detector,
            save_dir=Path(args.save_dir),
            use_mock_camera=args.mock,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stereo Vision Depth Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--camera", "-c", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--baseline", "-b", type=float, default=60.0, help="Stereo baseline in mm"
    )
    parser.add_argument(
        "--depth", "-d", type=float, default=1000.0, help="Initial simulated depth (mm)"
    )
    parser.add_argument(
        "--resolution", "-r", type=str, default="640x480", help="Resolution WxH"
    )
    parser.add_argument(
        "--scale", type=float, default=0.5, help="Display scale factor"
    )
    parser.add_argument(
        "--fast", "-F", action="store_true", help="Enable fast mode"
    )
    parser.add_argument(
        "--no-detector", action="store_true", help="Disable object detection"
    )
    parser.add_argument(
        "--save-dir", type=str, default="captures", help="Save directory"
    )
    parser.add_argument(
        "--mock", "-m", action="store_true",
        help="Use mock camera (synthetic data) instead of real hardware"
    )

    return parser.parse_args()


# ============================================================================
# UI Components
# ============================================================================


class InfoPanel:
    """Renders the information panel."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BG_COLOR = (40, 40, 40)
    TEXT_COLOR = (200, 200, 200)

    @staticmethod
    def render(
        width: int,
        camera_x: float,
        camera_y: float,
        sim_depth: float,
        fps: float,
        center_depth: float,
        stats: DepthStats | None,
        fast_mode: bool,
        process_ms: float,
        detections: list[tuple[Detection, float, tuple[float, float, float]]],
    ) -> npt.NDArray:
        """Create info panel image."""
        height = 200 + min(len(detections), 5) * 40
        panel = np.full((height, width, 3), InfoPanel.BG_COLOR, dtype=np.uint8)

        y = 25

        # Title
        mode_str = " [FAST]" if fast_mode else ""
        cv2.putText(panel, f"=== STEREO VISION{mode_str} ===", (10, y),
                    InfoPanel.FONT, 0.6, (100, 255, 100), 1)
        y += 30

        # FPS
        fps_color = (100, 255, 100) if fps >= 15 else (100, 100, 255)
        cv2.putText(panel, f"FPS: {fps:.1f} | Process: {process_ms:.1f}ms",
                    (10, y), InfoPanel.FONT, 0.5, fps_color, 1)
        y += 25

        # Center depth
        if center_depth > 0:
            cv2.putText(panel, f"CENTER: {center_depth:.0f}mm ({center_depth/10:.1f}cm)",
                        (10, y), InfoPanel.FONT, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(panel, "CENTER: --", (10, y), InfoPanel.FONT, 0.6, (100, 100, 100), 1)
        y += 30

        # Stats
        if stats:
            cv2.putText(panel, f"Range: {stats.min_mm:.0f}-{stats.max_mm:.0f}mm",
                        (10, y), InfoPanel.FONT, 0.5, (200, 200, 100), 1)
            y += 20
            cv2.putText(panel, f"Valid: {stats.valid_ratio*100:.1f}%",
                        (10, y), InfoPanel.FONT, 0.5, (200, 200, 100), 1)
            y += 25

        # Camera
        cv2.putText(panel, f"Pos: X={camera_x:+.1f} Y={camera_y:+.1f}mm",
                    (10, y), InfoPanel.FONT, 0.5, InfoPanel.TEXT_COLOR, 1)
        y += 20
        cv2.putText(panel, f"Sim depth: {sim_depth:.0f}mm",
                    (10, y), InfoPanel.FONT, 0.5, InfoPanel.TEXT_COLOR, 1)
        y += 25

        # Detections with depth
        if detections:
            cv2.putText(panel, f"=== DETECTED OBJECTS ({len(detections)}) ===",
                        (10, y), InfoPanel.FONT, 0.55, (255, 200, 100), 1)
            y += 25
            for det, depth_mm, pos_3d in detections[:5]:
                # Object label and depth prominently
                if depth_mm > 0:
                    depth_cm = depth_mm / 10
                    depth_m = depth_mm / 1000
                    cv2.putText(panel, f"{det.label.upper()}: {depth_mm:.0f}mm = {depth_cm:.1f}cm = {depth_m:.2f}m",
                                (10, y), InfoPanel.FONT, 0.5, (0, 255, 255), 1)
                    y += 18
                    # 3D position
                    x3d, y3d, z3d = pos_3d
                    cv2.putText(panel, f"  -> 3D: X={x3d:.0f} Y={y3d:.0f} Z={z3d:.0f}mm",
                                (10, y), InfoPanel.FONT, 0.4, (150, 255, 150), 1)
                else:
                    cv2.putText(panel, f"{det.label.upper()}: NO DEPTH",
                                (10, y), InfoPanel.FONT, 0.5, (100, 100, 100), 1)
                y += 22
        else:
            cv2.putText(panel, "No objects detected",
                        (10, y), InfoPanel.FONT, 0.5, (100, 100, 100), 1)

        return panel


class HelpOverlay:
    """Renders keyboard help overlay."""

    SHORTCUTS = [
        ("Arrows", "Move camera"),
        ("+/-", "Adjust depth"),
        ("Space", "Reset position"),
        ("C", "Cycle colormap"),
        ("D", "Toggle depth blend"),
        ("F", "Toggle fast mode"),
        ("V", "Toggle verbose stats"),
        ("H", "Toggle this help"),
        ("S", "Save frame"),
        ("Q/ESC", "Quit"),
    ]

    @staticmethod
    def render(width: int, height: int) -> npt.NDArray:
        """Create help overlay."""
        overlay = np.zeros((height, width, 3), dtype=np.uint8)

        # Background box
        pad = 50
        cv2.rectangle(overlay, (pad, pad), (width - pad, height - pad), (40, 40, 40), -1)
        cv2.rectangle(overlay, (pad, pad), (width - pad, height - pad), (100, 255, 100), 2)

        y = 90
        cv2.putText(overlay, "KEYBOARD SHORTCUTS", (70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        y += 40

        for key, desc in HelpOverlay.SHORTCUTS:
            cv2.putText(overlay, f"{key:10}", (70, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            cv2.putText(overlay, desc, (180, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 25

        return overlay


# ============================================================================
# Main Demo Class
# ============================================================================


class StereoVisionDemo:
    """Interactive stereo vision demonstration."""

    COLORMAPS = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_RAINBOW]
    COLORMAP_NAMES = ["JET", "HOT", "RAINBOW"]

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.camera: MockStereoCamera | RealStereoCamera | None = None
        self.depth_calc: DepthCalculator | None = None
        self.detector: DummyDetector | None = None

        # State
        self.running = True
        self.show_help = False
        self.show_depth_blend = True
        self.fast_mode = config.fast_mode
        self.verbose_mode = True  # Show detailed stats
        self.colormap_idx = 0

        # Stats
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_print_time = time.time()
        self.process_ms = 0.0

        # Mouse state
        self._depth_map: npt.NDArray | None = None

    def setup(self) -> None:
        """Initialize all components."""
        print("=" * 60)
        print("  KISHIRISU - Stereo Vision Demo")
        print("=" * 60)

        # Create stereo config
        print("\n[1/4] Configuring stereo system...")
        stereo_config = StereoConfig.from_camera(
            camera_index=self.config.camera_index,
            baseline_mm=self.config.baseline_mm,
            prefer_resolution=self.config.resolution,
            preset=self.config.preset,
        )

        # Initialize camera
        print("[2/4] Initializing camera...")
        if self.config.use_mock_camera:
            print("        Using MOCK camera (synthetic data)")
            self.camera = MockStereoCamera(
                config=stereo_config,
                camera_index=self.config.camera_index,
                simulated_depth_mm=self.config.simulated_depth_mm,
                add_noise=True,
            )
        else:
            print("        Using REAL camera")
            try:
                self.camera = RealStereoCamera(
                    config=stereo_config,
                    camera_index=self.config.camera_index,
                    mode=CaptureMode.SPLIT,
                )
            except RuntimeError as e:
                print(f"        [WARNING] Failed to open real camera: {e}")
                print("        Falling back to mock camera...")
                self.camera = MockStereoCamera(
                    config=stereo_config,
                    camera_index=self.config.camera_index,
                    simulated_depth_mm=self.config.simulated_depth_mm,
                    add_noise=True,
                )

        # Initialize depth calculator
        print("[3/4] Initializing depth calculator...")
        self.depth_calc = DepthCalculator(stereo_config)

        # Initialize detector
        print("[4/4] Initializing detector...")
        if self.config.enable_detector:
            self.detector = DummyDetector(label="ball")

        # Create save directory
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup window
        cv2.namedWindow("Stereo Vision Demo")
        cv2.setMouseCallback("Stereo Vision Demo", self._on_mouse)

        print("\nStarting... Press 'H' for help, 'Q' to quit.")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse clicks."""
        if event != cv2.EVENT_LBUTTONDOWN or self._depth_map is None:
            return

        scale = self.config.display_scale
        h, w = self._depth_map.shape
        panel_w = int(w * scale)
        panel_h = int(h * scale)

        # Check which panel was clicked
        if y < panel_h and x < panel_w:
            # Left camera or depth panel
            img_x = int(x / scale)
            img_y = int(y / scale)
        elif y >= panel_h and x < panel_w:
            # Depth panel
            img_x = int(x / scale)
            img_y = int((y - panel_h) / scale)
        else:
            return

        if self.depth_calc:
            depth = self.depth_calc.get_depth_at(self._depth_map, img_x, img_y)
            if depth > 0:
                x3d, y3d, z3d = self.depth_calc.pixel_to_3d(img_x, img_y, depth)
                print(f"\n>>> CLICK ({img_x},{img_y}): {depth:.0f}mm | 3D=({x3d:.0f},{y3d:.0f},{z3d:.0f}) <<<")
            else:
                print(f"\n>>> CLICK ({img_x},{img_y}): No valid depth <<<")

    def run(self) -> None:
        """Main demo loop."""
        if not self.camera or not self.depth_calc:
            raise RuntimeError("Setup not called")

        try:
            while self.running:
                self._process_frame()
                self._handle_input()
                self._update_fps()

        except KeyboardInterrupt:
            print("\nInterrupted")

    def _process_frame(self) -> None:
        """Capture and process one frame."""
        if not self.camera or not self.depth_calc:
            return

        # Capture
        left, right = self.camera.capture()

        # Compute depth
        t0 = time.time()
        if self.fast_mode:
            result = self.depth_calc.compute_fast(left, right)
        else:
            result = self.depth_calc.compute(left, right)
        self.process_ms = (time.time() - t0) * 1000
        self._depth_map = result.depth_map

        center_depth = result.at_center()

        # Detect objects and compute their depths
        detections_with_depth: list[tuple[Detection, float, tuple[float, float, float]]] = []
        if self.detector:
            raw_detections = self.detector.detect(left)
            for det in raw_detections:
                # Get depth at detection center
                depth_mm = self.depth_calc.get_depth_at(result.depth_map, det.x, det.y, window_size=11)
                # Compute 3D position
                pos_3d = self.depth_calc.pixel_to_3d(det.x, det.y, depth_mm)
                detections_with_depth.append((det, depth_mm, pos_3d))

                # Draw on left frame with depth info
                det.draw_on(left)
                if depth_mm > 0:
                    # Draw depth label on detection
                    depth_text = f"{depth_mm:.0f}mm ({depth_mm/10:.1f}cm)"
                    cv2.putText(left, depth_text, (det.x - 50, det.y + det.height // 2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Visualize depth
        depth_vis = self.depth_calc.visualize(
            result.depth_map, self.COLORMAPS[self.colormap_idx]
        )

        # Draw center crosshair
        cx, cy = self.camera.config.width // 2, self.camera.config.height // 2
        cv2.drawMarker(depth_vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        if center_depth > 0:
            cv2.putText(depth_vis, f"{center_depth:.0f}mm", (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw detection depths on depth visualization too
        for det, depth_mm, pos_3d in detections_with_depth:
            if depth_mm > 0:
                cv2.drawMarker(depth_vis, (det.x, det.y), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(depth_vis, f"{depth_mm:.0f}mm", (det.x + 10, det.y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create display
        display = self._compose_display(
            left, right, depth_vis, result.stats, center_depth, detections_with_depth
        )

        # Help overlay
        if self.show_help:
            help_img = HelpOverlay.render(display.shape[1], display.shape[0])
            display = cv2.addWeighted(display, 0.7, help_img, 0.3, 0)

        cv2.imshow("Stereo Vision Demo", display)

        # Periodic terminal output with detection depths
        if time.time() - self.last_print_time >= 0.5:
            self.last_print_time = time.time()
            print("\n" + "=" * 80)
            print(f"[CENTER DEPTH] {center_depth:.0f}mm = {center_depth/10:.1f}cm = {center_depth/1000:.2f}m" if center_depth > 0 else "[CENTER DEPTH] No valid depth")
            print(f"[PERFORMANCE] FPS: {self.fps:.1f} | Processing: {self.process_ms:.1f}ms | Mode: {'FAST' if self.fast_mode else 'NORMAL'}")
            
            # Verbose depth statistics
            if self.verbose_mode:
                stats = result.stats
                print(f"[DEPTH STATS]")
                print(stats.format_verbose())
            
            # Detection output
            if detections_with_depth:
                print(f"[DETECTIONS] Found {len(detections_with_depth)} object(s):")
                for i, (det, depth_mm, pos_3d) in enumerate(detections_with_depth[:5], 1):
                    if depth_mm > 0:
                        x3d, y3d, z3d = pos_3d
                        print(f"  {i}. {det.label.upper()} @ ({det.x},{det.y}) -> {depth_mm:.0f}mm | 3D=({x3d:.0f},{y3d:.0f},{z3d:.0f})")
                    else:
                        print(f"  {i}. {det.label.upper()} @ ({det.x},{det.y}) -> NO DEPTH")
            print("=" * 80)

    def _compose_display(
        self,
        left: npt.NDArray,
        right: npt.NDArray,
        depth: npt.NDArray,
        stats: DepthStats,
        center_depth: float,
        detections: list[tuple[Detection, float, tuple[float, float, float]]],
    ) -> npt.NDArray:
        """Compose the display layout."""
        if not self.camera:
            raise RuntimeError("Camera not initialized")

        h, w = left.shape[:2]

        # Label frames
        left_labeled = left.copy()
        right_labeled = right.copy()
        cv2.putText(left_labeled, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(right_labeled, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Depth display
        if self.show_depth_blend:
            depth_display = cv2.addWeighted(left, 0.4, depth, 0.6, 0)
        else:
            depth_display = depth
        cv2.putText(depth_display, "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Info panel - handle mock vs real camera properties
        if isinstance(self.camera, MockStereoCamera):
            camera_x = self.camera.current_x
            camera_y = self.camera.current_y
            sim_depth = self.camera.simulated_depth
        else:
            camera_x = 0.0
            camera_y = 0.0
            sim_depth = 0.0  # Not applicable for real camera

        info = InfoPanel.render(
            width=w,
            camera_x=camera_x,
            camera_y=camera_y,
            sim_depth=sim_depth,
            fps=self.fps,
            center_depth=center_depth,
            stats=stats,
            fast_mode=self.fast_mode,
            process_ms=self.process_ms,
            detections=detections,
        )
        info_resized = cv2.resize(info, (w, h))

        # Compose grid
        top = np.hstack([left_labeled, right_labeled])
        bottom = np.hstack([depth_display, info_resized])
        display = np.vstack([top, bottom])

        # Scale
        if self.config.display_scale != 1.0:
            new_w = int(display.shape[1] * self.config.display_scale)
            new_h = int(display.shape[0] * self.config.display_scale)
            display = cv2.resize(display, (new_w, new_h))

        return display

    def _handle_input(self) -> None:
        """Process keyboard input."""
        if not self.camera:
            return

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            self.running = False

        elif key == ord("h"):
            self.show_help = not self.show_help

        elif key == ord("d"):
            self.show_depth_blend = not self.show_depth_blend

        elif key == ord("f"):
            self.fast_mode = not self.fast_mode
            print(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")

        elif key == ord("c"):
            self.colormap_idx = (self.colormap_idx + 1) % len(self.COLORMAPS)
            print(f"Colormap: {self.COLORMAP_NAMES[self.colormap_idx]}")

        elif key == ord("v"):
            self.verbose_mode = not self.verbose_mode
            print(f"Verbose mode: {'ON' if self.verbose_mode else 'OFF'}")

        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = self.config.save_dir / f"capture_{timestamp}.png"
            # Would save display here
            print(f"Saved: {path}")

        # Mock camera only controls
        elif isinstance(self.camera, MockStereoCamera):
            if key == ord(" "):
                self.camera.reset_position()

            elif key in (ord("="), ord("+")):
                self.camera.set_simulated_depth(self.camera.simulated_depth + 100)

            elif key == ord("-"):
                self.camera.set_simulated_depth(max(200, self.camera.simulated_depth - 100))

            # Arrow keys (camera position)
            elif key in (81, 2):  # Left
                self.camera.move_relative(-5, 0)
            elif key in (83, 3):  # Right
                self.camera.move_relative(5, 0)
            elif key in (82, 0):  # Up
                self.camera.move_relative(0, -5)
            elif key in (84, 1):  # Down
                self.camera.move_relative(0, 5)

    def _update_fps(self) -> None:
        """Update FPS counter."""
        self.frame_count += 1
        elapsed = time.time() - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = time.time()

    def cleanup(self) -> None:
        """Release resources."""
        print("Releasing resources...")
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Done!")


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = DemoConfig.from_args(args)

    demo = StereoVisionDemo(config)

    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        demo.running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        demo.setup()
        demo.run()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
