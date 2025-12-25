"""
Unit tests for mock stereo camera module.
"""

import numpy as np
import pytest

from stereo_vision import (
    StereoConfig,
    Resolution,
)
from stereo_vision.mock_camera import (
    MockStereoCamera,
    ProceduralSceneGenerator,
    SceneObject,
    SceneInfo,
    ObjectShape,
    Scene,
    StereoSynthesizer,
)


class TestSceneObject:
    """Tests for SceneObject dataclass."""

    def test_creation(self) -> None:
        obj = SceneObject(
            x=100, y=200, width=50, height=50,
            depth_mm=1000.0, shape=ObjectShape.CIRCLE, label="ball"
        )
        assert obj.x == 100
        assert obj.y == 200
        assert obj.depth_mm == 1000.0

    def test_radius_property(self) -> None:
        obj = SceneObject(x=0, y=0, width=100, height=80, depth_mm=500.0)
        assert obj.radius == 40  # min(100, 80) // 2

    def test_str_representation(self) -> None:
        obj = SceneObject(
            x=50, y=50, width=30, height=30,
            depth_mm=1500.0, shape=ObjectShape.RECTANGLE, label="box"
        )
        s = str(obj)
        assert "box" in s
        assert "RECTANGLE" in s
        assert "1500" in s


class TestSceneInfo:
    """Tests for SceneInfo metadata."""

    def test_creation(self) -> None:
        info = SceneInfo(
            frame_number=42,
            object_count=5,
            depth_range=(300.0, 5000.0),
            has_occlusion=True,
            complexity=7,
            description="Test scene"
        )
        assert info.frame_number == 42
        assert info.object_count == 5
        assert info.has_occlusion is True

    def test_format_verbose(self) -> None:
        info = SceneInfo(
            frame_number=10,
            object_count=3,
            depth_range=(500.0, 3000.0),
            has_occlusion=False,
            complexity=5,
            description="Scene with 3 circles"
        )
        verbose = info.format_verbose()
        assert "Frame: 10" in verbose
        assert "Objects: 3" in verbose
        assert "500" in verbose
        assert "3000" in verbose


class TestProceduralSceneGenerator:
    """Tests for procedural scene generation."""

    @pytest.fixture
    def generator(self) -> ProceduralSceneGenerator:
        return ProceduralSceneGenerator(
            base_depth_mm=1000.0,
            min_depth_mm=300.0,
            max_depth_mm=5000.0,
            num_objects=4,
        )

    def test_generate_returns_scene(self, generator: ProceduralSceneGenerator) -> None:
        scene = generator.generate(640, 480, frame=1)
        assert isinstance(scene, Scene)
        assert scene.depth_map is not None
        assert scene.image is not None
        assert scene.info is not None

    def test_depth_map_shape(self, generator: ProceduralSceneGenerator) -> None:
        scene = generator.generate(640, 480, frame=1)
        assert scene.depth_map.shape == (480, 640)

    def test_image_shape(self, generator: ProceduralSceneGenerator) -> None:
        scene = generator.generate(640, 480, frame=1)
        assert scene.image.shape == (480, 640, 3)

    def test_objects_created(self, generator: ProceduralSceneGenerator) -> None:
        scene = generator.generate(640, 480, frame=1)
        # num_objects + center object
        assert len(scene.objects) >= 4

    def test_depth_values_in_range(self, generator: ProceduralSceneGenerator) -> None:
        scene = generator.generate(640, 480, frame=1)
        valid = scene.depth_map[scene.depth_map > 0]
        assert np.min(valid) >= 200  # Some tolerance
        assert np.max(valid) <= 6000

    def test_scene_info_populated(self, generator: ProceduralSceneGenerator) -> None:
        scene = generator.generate(640, 480, frame=5)
        assert scene.info is not None
        assert scene.info.frame_number == 5
        assert scene.info.object_count == len(scene.objects)

    def test_different_frames_produce_different_scenes(
        self, generator: ProceduralSceneGenerator
    ) -> None:
        scene1 = generator.generate(640, 480, frame=1)
        scene2 = generator.generate(640, 480, frame=100)
        # Objects should be in different positions
        assert scene1.objects[0].x != scene2.objects[0].x or \
               scene1.objects[0].y != scene2.objects[0].y

    def test_diverse_object_shapes(self, generator: ProceduralSceneGenerator) -> None:
        scene = generator.generate(640, 480, frame=1)
        shapes = {obj.shape for obj in scene.objects}
        # Should have multiple shape types
        assert len(shapes) >= 2

    def test_set_base_depth(self, generator: ProceduralSceneGenerator) -> None:
        generator.set_base_depth(2000.0)
        assert generator.base_depth == 2000.0


class TestStereoSynthesizer:
    """Tests for stereo pair synthesis."""

    @pytest.fixture
    def synthesizer(self) -> StereoSynthesizer:
        config = StereoConfig(resolution=Resolution(320, 240))
        return StereoSynthesizer(config)

    def test_synthesize_right_shape(self, synthesizer: StereoSynthesizer) -> None:
        left = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 1000.0, dtype=np.float32)
        
        right = synthesizer.synthesize_right(left, depth)
        
        assert right.shape == left.shape

    def test_synthesize_right_dtype(self, synthesizer: StereoSynthesizer) -> None:
        left = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 1000.0, dtype=np.float32)
        
        right = synthesizer.synthesize_right(left, depth)
        
        assert right.dtype == np.uint8


class TestMockStereoCamera:
    """Tests for MockStereoCamera class."""

    @pytest.fixture
    def camera(self) -> MockStereoCamera:
        config = StereoConfig(resolution=Resolution(320, 240))
        return MockStereoCamera(config, simulated_depth_mm=1000.0)

    def test_capture_returns_pair(self, camera: MockStereoCamera) -> None:
        left, right = camera.capture()
        assert left.shape == (240, 320, 3)
        assert right.shape == (240, 320, 3)

    def test_left_right_different(self, camera: MockStereoCamera) -> None:
        left, right = camera.capture()
        # Images should not be identical (right is shifted)
        assert not np.array_equal(left, right)

    def test_simulated_depth_property(self, camera: MockStereoCamera) -> None:
        assert camera.simulated_depth == 1000.0

    def test_set_simulated_depth(self, camera: MockStereoCamera) -> None:
        camera.set_simulated_depth(2000.0)
        assert camera.simulated_depth == 2000.0

    def test_set_simulated_depth_invalid(self, camera: MockStereoCamera) -> None:
        with pytest.raises(ValueError):
            camera.set_simulated_depth(-100)

    def test_is_opened(self, camera: MockStereoCamera) -> None:
        assert camera.is_opened() is True

    def test_context_manager(self) -> None:
        config = StereoConfig(resolution=Resolution(320, 240))
        with MockStereoCamera(config) as camera:
            left, right = camera.capture()
            assert left is not None

    def test_get_info(self, camera: MockStereoCamera) -> None:
        info = camera.get_info()
        assert info["type"] == "MockStereoCamera"
        assert info["mode"] == "synthetic"

    def test_position_movement(self, camera: MockStereoCamera) -> None:
        camera.reset_position()
        assert camera.current_x == 0.0
        assert camera.current_y == 0.0
        
        camera.move_relative(10, 5)
        assert camera.current_x == 10.0
        assert camera.current_y == 5.0
