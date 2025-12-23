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
- **Mock stereo** - Single webcam generates synthetic stereo pairs
- **Modular detection** - Plug-in interface for YOLO integration
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

### YOLO Integration

For your friend implementing YOLO detection:

```python
from stereo_vision import Detection, BoundingBox, Detector
from ultralytics import YOLO

class YOLODetector:
    """YOLO-based object detector."""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def detect(self, frame) -> list[Detection]:
        results = self.model(frame)
        detections = []
        
        for box in results[0].boxes:
            x, y, w, h = box.xywh[0].cpu().numpy()
            detections.append(Detection(
                label=self.model.names[int(box.cls)],
                bbox=BoundingBox(cx=int(x), cy=int(y), w=int(w), h=int(h)),
                confidence=float(box.conf),
            ))
        
        return detections
```

## Demo Controls

| Key | Action |
|-----|--------|
| Arrow keys | Move camera position |
| +/- | Adjust simulated depth |
| Space | Reset position |
| C | Cycle colormap |
| D | Toggle depth overlay |
| F | Toggle fast mode |
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
