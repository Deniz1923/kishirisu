"""
Unit tests for stereo vision configuration module.
"""

import json
import pytest

from stereo_vision import (
    Resolution,
    QualityPreset,
    SGBMParams,
    StereoConfig,
)


class TestResolution:
    """Tests for Resolution namedtuple."""

    def test_creation(self) -> None:
        res = Resolution(1920, 1080)
        assert res.width == 1920
        assert res.height == 1080

    def test_aspect_ratio(self) -> None:
        res = Resolution(1920, 1080)
        assert pytest.approx(res.aspect_ratio, rel=0.01) == 16 / 9

    def test_scaled(self) -> None:
        res = Resolution(1280, 720)
        scaled = res.scaled(0.5)
        assert scaled.width == 640
        assert scaled.height == 360

    def test_str(self) -> None:
        res = Resolution(640, 480)
        assert str(res) == "640x480"

    def test_parse(self) -> None:
        res = Resolution.parse("1280x720")
        assert res.width == 1280
        assert res.height == 720


class TestSGBMParams:
    """Tests for SGBMParams configuration."""

    def test_defaults(self) -> None:
        params = SGBMParams()
        assert params.num_disparities == 64
        assert params.block_size == 11

    def test_auto_p1_p2(self) -> None:
        params = SGBMParams(block_size=7)
        p1 = params.get_p1()
        p2 = params.get_p2()
        # P1 = 8 * 3 * block_size^2 = 8 * 3 * 49 = 1176
        assert p1 == 8 * 3 * 49
        # P2 = 32 * 3 * block_size^2 = 32 * 3 * 49 = 4704
        assert p2 == 32 * 3 * 49

    def test_preset_fast(self) -> None:
        params = SGBMParams.for_preset(QualityPreset.FAST)
        assert params.num_disparities == 48
        assert params.block_size == 5

    def test_preset_quality(self) -> None:
        params = SGBMParams.for_preset(QualityPreset.QUALITY)
        assert params.num_disparities == 128
        assert params.block_size == 11


class TestStereoConfig:
    """Tests for StereoConfig dataclass."""

    def test_default_creation(self) -> None:
        config = StereoConfig()
        assert config.width == 640
        assert config.height == 480
        assert config.baseline_mm == 60.0

    def test_focal_estimation(self) -> None:
        config = StereoConfig(resolution=Resolution(1280, 720))
        # Should auto-estimate ~70° HFOV
        assert config.focal == pytest.approx(1280 * 0.82)

    def test_explicit_focal(self) -> None:
        config = StereoConfig(focal_length_px=1000.0)
        assert config.focal == 1000.0

    def test_principal_point_default(self) -> None:
        config = StereoConfig(resolution=Resolution(640, 480))
        assert config.cx == 320.0
        assert config.cy == 240.0

    def test_depth_factor(self) -> None:
        config = StereoConfig(
            resolution=Resolution(640, 480),
            focal_length_px=500.0,
            baseline_mm=100.0,
        )
        assert config.depth_factor == 500.0 * 100.0

    def test_invalid_resolution(self) -> None:
        with pytest.raises(ValueError, match="Invalid resolution"):
            StereoConfig(resolution=Resolution(-1, 480))

    def test_invalid_baseline(self) -> None:
        with pytest.raises(ValueError, match="Baseline must be positive"):
            StereoConfig(baseline_mm=0)

    def test_invalid_num_disparities(self) -> None:
        with pytest.raises(ValueError, match="divisible by 16"):
            StereoConfig(sgbm=SGBMParams(num_disparities=50))

    def test_invalid_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_size must be odd"):
            StereoConfig(sgbm=SGBMParams(block_size=6))

    def test_scaled(self) -> None:
        config = StereoConfig(resolution=Resolution(1280, 720))
        scaled = config.scaled(0.5)
        assert scaled.resolution == Resolution(640, 360)
        assert scaled.focal == pytest.approx(config.focal * 0.5)

    def test_for_preset(self) -> None:
        config = StereoConfig.for_preset(QualityPreset.FAST)
        assert config.sgbm.num_disparities == 48

    def test_to_dict(self) -> None:
        config = StereoConfig()
        d = config.to_dict()
        assert d["resolution"] == [640, 480]
        assert d["baseline_mm"] == 60.0
        assert "sgbm" in d

    def test_to_json(self) -> None:
        config = StereoConfig()
        j = config.to_json()
        parsed = json.loads(j)
        assert parsed["resolution"] == [640, 480]

    def test_from_dict_roundtrip(self) -> None:
        original = StereoConfig(
            resolution=Resolution(1280, 720),
            baseline_mm=70.0,
            min_depth_mm=300.0,
        )
        d = original.to_dict()
        restored = StereoConfig.from_dict(d)
        
        assert restored.resolution == original.resolution
        assert restored.baseline_mm == original.baseline_mm
        assert restored.min_depth_mm == original.min_depth_mm

    def test_from_json_roundtrip(self) -> None:
        original = StereoConfig()
        j = original.to_json()
        restored = StereoConfig.from_json(j)
        
        assert restored.resolution == original.resolution
        assert restored.baseline_mm == original.baseline_mm

    def test_camera_matrix_shape(self) -> None:
        config = StereoConfig()
        K = config.get_camera_matrix()
        assert K.shape == (3, 3)
        assert K[0, 0] == config.focal  # fx
        assert K[1, 1] == config.focal  # fy
        assert K[0, 2] == config.cx     # cx
        assert K[1, 2] == config.cy     # cy
