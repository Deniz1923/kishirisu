# Kishirisu 🏐

Stereo vision depth estimation for a ball catcher robot.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

## Quick Start

```bash
# Install dependencies
uv sync

# Run the demo
uv run python main.py

# With options
uv run python main.py --fast --resolution 1280x720
```

## Features

- **Real-time depth** - SGBM stereo matching with quality presets
- **Verbose statistics** - Depth zones, percentiles, confidence scoring
- **Mock stereo** - Synthetic stereo pairs with diverse objects
- **YOLO-ready detection** - `BallDetector` interface for easy integration
- **Interactive demo** - Keyboard controls, mouse depth queries

## Architecture

```
stereo_vision/
├── config.py      # StereoConfig, Resolution, QualityPreset
├── capture.py     # StereoCapture protocol, CameraPosition
├── mock_camera.py # MockStereoCamera (single webcam → stereo)
├── depth.py       # DepthCalculator, DepthResult, DepthStats
└── detection.py   # Detection, BoundingBox, Detector protocol
```

## Usage

### Basic Depth Calculation

```python
from stereo_vision import StereoConfig, MockStereoCamera, DepthCalculator

# Auto-detect camera
config = StereoConfig.from_camera(baseline_mm=65.0)

with MockStereoCamera(config) as camera:
    left, right = camera.capture()
    
    calc = DepthCalculator(config)
    result = calc.compute(left, right)
    
    print(f"Center depth: {result.at_center():.0f}mm")
    print(f"Valid pixels: {result.stats.valid_ratio:.1%}")
```

### Quality Presets

```python
from stereo_vision import StereoConfig, QualityPreset

# Fast for real-time
config = StereoConfig.for_preset(QualityPreset.FAST)

# High quality for accuracy
config = StereoConfig.for_preset(QualityPreset.QUALITY)
```

### YOLO Ball Detection Integration

> **For your friend implementing ball detection** 🏀

The `BallDetector` abstract class provides a clear interface. Create a new file `yolo_detector.py`:

```python
from ultralytics import YOLO
from stereo_vision import BallDetector, Detection, BoundingBox

class YOLOBallDetector(BallDetector):
    """YOLO-based ball detector."""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.ball_class_id = 32  # COCO sports ball class
    
    def detect_balls(self, frame) -> list[Detection]:
        """Run YOLO inference and return ball detections."""
        results = self.model(frame, verbose=False)
        detections = []
        
        for box in results[0].boxes:
            if int(box.cls) == self.ball_class_id:
                x, y, w, h = box.xywh[0].cpu().numpy()
                detections.append(Detection(
                    label="ball",
                    bbox=BoundingBox(cx=int(x), cy=int(y), w=int(w), h=int(h)),
                    confidence=float(box.conf),
                ))
        
        return detections
```

**Using with depth calculation:**

```python
from stereo_vision import DepthCalculator, StereoConfig
from yolo_detector import YOLOBallDetector

# Initialize
detector = YOLOBallDetector("path/to/your/model.pt")
calc = DepthCalculator(config)

# Detect and get depth
detections = detector.detect_balls(left_frame)
result = calc.compute(left, right)

for det in detections:
    depth_mm = calc.get_depth_at(result.depth_map, det.x, det.y)
    x3d, y3d, z3d = calc.pixel_to_3d(det.x, det.y, depth_mm)
    print(f"Ball at 3D position: ({x3d:.0f}, {y3d:.0f}, {z3d:.0f})mm")
```

**Interface contract:**
| Method | Required | Description |
|--------|----------|-------------|
| `detect_balls(frame)` | ✅ Yes | Returns `list[Detection]` |
| `detect(frame)` | Auto | Alias for `detect_balls` |
| `detect_balls_batch(frames)` | Optional | Batch inference |
| `warmup()` | Optional | GPU warmup |


## Demo Controls

| Key | Action |
|-----|--------|
| Arrow keys | Move camera position |
| +/- | Adjust simulated depth |
| Space | Reset position |
| C | Cycle colormap |
| D | Toggle depth overlay |
| F | Toggle fast mode |
| V | Toggle verbose stats |
| H | Toggle help |
| S | Save frame |
| Q/ESC | Quit |
| Click | Query depth at point |

## Configuration

```python
from stereo_vision import StereoConfig, Resolution, SGBMParams

config = StereoConfig(
    resolution=Resolution(1280, 720),
    baseline_mm=65.0,           # Distance between cameras
    focal_length_px=1050.0,     # Or None to auto-estimate
    min_depth_mm=200.0,         # Minimum valid depth
    max_depth_mm=5000.0,        # Maximum valid depth
    sgbm=SGBMParams(
        num_disparities=128,    # Search range (÷16)
        block_size=11,          # Match window (odd, ≥5)
    ),
)
```

## Mock Camera

Since only one webcam is available, `MockStereoCamera` generates synthetic stereo:

1. Captures single frame → "left" image
2. Shifts horizontally based on simulated depth → "right" image  
3. Adds sensor noise for realism

This allows developing the depth pipeline before hardware arrives.

## Requirements

- Python 3.11+
- OpenCV 4.8+
- NumPy 1.26+

```bash
uv sync
```

## License

MIT
