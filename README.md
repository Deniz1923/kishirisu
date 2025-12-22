# Kishirisu - Ball Catcher Robot Stereo Vision

Stereo vision depth calculation system for a ball catcher robot using OpenCV.

## Quick Start

```bash
# Install dependencies
uv sync

# Run with auto-detection (uses your camera's native resolution/FPS)
uv run python main.py

# Or specify preferred resolution
uv run python main.py --resolution 1280x720
```

## Architecture

```
stereo_vision/
├── __init__.py      # Package exports
├── config.py        # StereoConfig with auto-detection
├── capture.py       # Abstract StereoCapture interface
├── mock_camera.py   # MockStereoCamera (single webcam → stereo pair)
├── depth.py         # DepthCalculator (SGBM stereo matching)
└── detection.py     # ObjectDetector interface (for YOLO integration)
```

## Usage

### Auto-Detection (Recommended)

```python
from stereo_vision import StereoConfig, MockStereoCamera, DepthCalculator

# Auto-detect camera resolution and FPS
config = StereoConfig.from_camera(camera_index=0, baseline_mm=60.0)
print(f"Detected: {config.resolution} @ {config.fps} FPS")

# Or use the convenience method
camera = MockStereoCamera.create_auto(camera_index=0)
print(f"Using: {camera.actual_resolution} @ {camera.actual_fps} FPS")

# Initialize depth calculator with detected config
depth_calc = DepthCalculator(camera.config)

# Capture and compute depth
left, right = camera.capture()
depth_map = depth_calc.compute(left, right)
depth_mm = depth_calc.get_depth_at(depth_map, x=320, y=240)

camera.release()
```

### Manual Configuration

```python
from stereo_vision import StereoConfig, MockStereoCamera, DepthCalculator

# Manual configuration (e.g., for testing)
config = StereoConfig(
    resolution=(1920, 1080),
    fps=30.0,
    baseline_mm=60.0,
    focal_length_px=1575.0,  # Or None to auto-estimate
)

camera = MockStereoCamera(config, camera_index=0)
```

## YOLO Integration (For Your Friend)

The detection interface is designed for easy YOLO integration:

```python
from stereo_vision import ObjectDetector, Detection
from ultralytics import YOLO

class YOLODetector(ObjectDetector):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for r in results[0].boxes:
            x, y, w, h = r.xywh[0].cpu().numpy()
            detections.append(Detection(
                label=self.model.names[int(r.cls)],
                x=int(x), y=int(y),
                width=int(w), height=int(h),
                confidence=float(r.conf)
            ))
        return detections
```

## Demo Controls

| Key | Action |
|-----|--------|
| Arrow keys | Move camera X/Y position |
| +/- | Adjust simulated depth |
| Space | Reset camera position |
| D | Toggle depth overlay |
| S | Save frame |
| Q/ESC | Quit |

## Mock Camera Mode

Since only one webcam is available, `MockStereoCamera` generates synthetic stereo pairs by:
1. Capturing single frame from webcam (→ "left" image)
2. Shifting horizontally based on simulated depth (→ "right" image)
3. Adding slight noise for realism

This allows developing and testing the depth pipeline before hardware is ready.

## Configuration

Key parameters in `StereoConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `baseline_mm` | 60.0 | Distance between cameras (mm) |
| `focal_length_px` | 640.0 | Camera focal length (pixels) |
| `resolution` | (640, 480) | Camera resolution |
| `num_disparities` | 64 | Disparity search range |
| `block_size` | 11 | SGBM block size |

## Dependencies

- Python 3.12+
- OpenCV 4.8+
- NumPy 1.26+
