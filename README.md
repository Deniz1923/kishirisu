# Kishirisu 🏐

Stereo vision depth estimation for a ball catcher robot.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.8+-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

## Quick Start

```bash
# Install dependencies
uv sync

# Run with mock camera (synthetic data for development)
uv run python main.py --mock

# Run with real stereo camera
uv run python main.py

# With options
uv run python main.py --mock --fast --resolution 1280x720
```

## Features

- **Accurate depth** - WLS edge-preserving filter, left-right consistency, temporal smoothing
- **Real-time modes** - SGBM stereo matching with quality presets (FAST/BALANCED/QUALITY)
- **Verbose statistics** - Depth zones, percentiles, confidence scoring
- **Mock stereo** - Synthetic stereo pairs with diverse objects
- **YOLO-ready detection** - `BallDetector` interface for easy integration
- **Interactive demo** - Keyboard controls, mouse depth queries

## Architecture

```
stereo_vision/
├── config.py       # StereoConfig, DepthFilterParams, QualityPreset
├── capture.py      # StereoCapture protocol, CameraPosition
├── mock_camera.py  # MockStereoCamera (synthetic stereo for testing)
├── real_camera.py  # RealStereoCamera (hardware capture)
├── depth.py        # DepthCalculator, DepthResult, DepthStats
└── detection.py    # Detection, BoundingBox, Detector protocol
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

| Key | Action | Mode |
|-----|--------|------|
| Arrow keys | Move camera position | Mock only |
| +/- | Adjust simulated depth | Mock only |
| Space | Reset position | Mock only |
| C | Cycle colormap | All |
| D | Toggle depth overlay | All |
| F | Toggle fast mode | All |
| V | Toggle verbose stats | All |
| H | Toggle help | All |
| S | Save frame | All |
| Q/ESC | Quit | All |
| Click | Query depth at point | All |

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

### Depth Filtering (Accuracy)

Fine-tune depth accuracy with `DepthFilterParams`:

```python
from stereo_vision import StereoConfig, DepthFilterParams

config = StereoConfig(
    baseline_mm=65.0,
    depth_filter=DepthFilterParams(
        use_wls_filter=True,    # Edge-preserving WLS smoothing
        wls_lambda=8000.0,      # Higher = smoother (8000-16000)
        wls_sigma=1.5,          # Higher = more edge-preserving
        left_right_check=True,  # Validate with right-to-left match
        lr_threshold=1,         # Max disparity difference (pixels)
        use_bilateral=False,    # Pre-filter input images
        temporal_alpha=0.2,     # Frame smoothing (0=off, 0.3=moderate)
    ),
)
```

> **Note**: WLS filter requires `opencv-contrib-python`. Install with:
> ```bash
> uv add opencv-contrib-python
> ```

## Real Camera Setup

The `RealStereoCamera` class supports two capture modes:

### Side-by-Side Stereo (Split Mode)

For cameras with stereo lens adapters that output a combined side-by-side image:

```python
from stereo_vision import RealStereoCamera, StereoConfig, CaptureMode

config = StereoConfig.from_camera(camera_index=0, baseline_mm=65.0)
camera = RealStereoCamera(config, camera_index=0, mode=CaptureMode.SPLIT)

left, right = camera.capture()
camera.release()
```

### Dual Camera Setup

For two separate USB cameras:

```python
from stereo_vision import RealStereoCamera, CaptureMode

camera = RealStereoCamera(
    config,
    camera_index=0,          # Left camera
    mode=CaptureMode.DUAL,
    right_camera_index=1,    # Right camera
)
```

### Fallback Behavior

The demo app automatically falls back to mock camera if real camera fails to open.

## Mock Camera

For development without hardware, `MockStereoCamera` generates synthetic stereo:

1. Generates procedural scene with depth map
2. Creates left image from scene
3. Shifts horizontally based on depth → right image  
4. Adds sensor noise for realism

```bash
# Always use mock camera
uv run python main.py --mock
```

### Simulation Modes

The mock camera supports two modes for generating synthetic data:

#### 1. Procedural Mode (Default)
Generates a random scene with:
- Floor plane with depth gradient
- 8+ objects of various shapes (circles, rectangles, ellipses)
- Moving target center
- Random motion and colors

#### 2. Ball Throw Simulation (`--ball-throw`)
Simulates a ball continually being thrown at the camera. Ideal for testing tracking algorithms.
- **Physics-based**: Gravity, parabolic trajectory, perspective scaling.
- **Controls**: Press **'T'** to trigger a new random throw.
- **Usage**:
  ```bash
  uv run python main.py --mock --ball-throw
  ```

---

## How Stereo Vision Works (Beginner's Guide) 🎓

If you are new to depth estimation, here is the concept in plain English.

### 1. The "Two Eyes" Concept
Stereo vision mimics human eyes. When you look at an object:
- Your **Left Eye** sees it at position $X_L$.
- Your **Right Eye** sees it at position $X_R$.
- Because your eyes are apart, there is a shift called **Disparity**.

Try this: Hold your finger in front of your face. Close one eye, then the other. Your finger seems to jump left/right. That "jump" is the disparity.

### 2. The Golden Rule
> **Closer objects have BIGGER disparity. Farther objects have SMALLER disparity.**

- **Close finger**: Jumps a lot (large shift).
- **Far mountain**: Doesn't jump at all (zero shift).

### 3. The Formula
To turn that pixel shift (disparity) into real distance (depth), we use three numbers:

$$ Depth = \frac{\text{Focal Length} \times \text{Baseline}}{\text{Disparity}} $$

- **Baseline ($B$)**: Distance between the two cameras (mm). *Fixed content.*
- **Focal Length ($f$)**: "Zoom" level of the lens (pixels). *Fixed constant.*
- **Disparity ($d$)**: Difference in pixel position ($x_{left} - x_{right}$). *Calculated per pixel.*

### Summary for this project
1. The camera captures two images.
2. The software finds matching pixels between Left and Right images.
3. It calculates the **Disparity** (pixels) for every point.
4. It divides the constant $(f \times B)$ by disparity to get **Depth** (mm).


## Requirements

- Python 3.11+
- OpenCV 4.8+
- NumPy 1.26+

```bash
uv sync
```

## License

MIT
