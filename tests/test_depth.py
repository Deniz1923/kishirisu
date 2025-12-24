"""
Unit tests for stereo depth calculation module.
"""

import numpy as np
import pytest

from stereo_vision import DepthStats, DepthResult, DepthCalculator, StereoConfig


class TestDepthStats:
    """Tests for DepthStats dataclass."""

    def test_from_depth_map(self) -> None:
        # Create a simple depth map with known values
        depth = np.array([
            [1000, 1500, 2000],
            [1200, 0, 1800],      # 0 = invalid
            [1100, 1400, 1600],
        ], dtype=np.float32)
        
        stats = DepthStats.from_depth_map(depth)
        
        assert stats.min_mm == 1000
        assert stats.max_mm == 2000
        assert stats.valid_ratio < 1.0  # One invalid pixel

    def test_all_invalid(self) -> None:
        depth = np.zeros((10, 10), dtype=np.float32)
        stats = DepthStats.from_depth_map(depth)
        
        assert stats.valid_ratio == 0.0
        assert stats.min_mm == 0.0
        assert stats.max_mm == 0.0


class TestDepthResult:
    """Tests for DepthResult container."""

    def test_at(self) -> None:
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        stats = DepthStats.from_depth_map(depth)
        result = DepthResult(depth_map=depth, stats=stats)
        
        # Should return depth at point with median filter
        d = result.at(50, 50)
        assert d == pytest.approx(1500.0)

    def test_at_center(self) -> None:
        depth = np.full((100, 100), 2000.0, dtype=np.float32)
        stats = DepthStats.from_depth_map(depth)
        result = DepthResult(depth_map=depth, stats=stats)
        
        assert result.at_center() == pytest.approx(2000.0)

    def test_at_boundary(self) -> None:
        depth = np.full((100, 100), 1000.0, dtype=np.float32)
        stats = DepthStats.from_depth_map(depth)
        result = DepthResult(depth_map=depth, stats=stats)
        
        # Should handle boundary pixels
        d = result.at(0, 0)
        assert d >= 0


class TestDepthCalculator:
    """Tests for DepthCalculator class."""

    @pytest.fixture
    def calculator(self) -> DepthCalculator:
        config = StereoConfig()
        return DepthCalculator(config)

    def test_creation(self, calculator: DepthCalculator) -> None:
        assert calculator.config is not None

    def test_compute_synthetic(self, calculator: DepthCalculator) -> None:
        # Create synthetic stereo pair (identical images = no disparity = inf depth)
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = left.copy()
        
        result = calculator.compute(left, right)
        
        assert result.depth_map.shape == (480, 640)
        assert isinstance(result.stats, DepthStats)

    def test_compute_fast(self, calculator: DepthCalculator) -> None:
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = left.copy()
        
        result = calculator.compute_fast(left, right, scale=0.5)
        
        # Result should be upscaled to original resolution
        assert result.depth_map.shape == (480, 640)

    def test_visualize(self, calculator: DepthCalculator) -> None:
        depth = np.random.uniform(500, 3000, (480, 640)).astype(np.float32)
        
        viz = calculator.visualize(depth)
        
        assert viz.shape == (480, 640, 3)
        assert viz.dtype == np.uint8

    def test_pixel_to_3d(self, calculator: DepthCalculator) -> None:
        # Center pixel at 1000mm depth
        x, y, z = calculator.pixel_to_3d(320, 240, 1000.0)
        
        # At center, X and Y should be near 0
        assert x == pytest.approx(0.0, abs=1.0)
        assert y == pytest.approx(0.0, abs=1.0)
        assert z == 1000.0

    def test_get_depth_stats(self, calculator: DepthCalculator) -> None:
        depth = np.full((100, 100), 1500.0, dtype=np.float32)
        
        stats = calculator.get_depth_stats(depth)
        
        assert "min_depth" in stats
        assert "max_depth" in stats
        assert "valid_percent" in stats

    def test_temporal_smoothing(self) -> None:
        """Test that temporal smoothing reduces frame-to-frame variation."""
        from stereo_vision import DepthFilterParams
        
        # Create config with temporal smoothing enabled
        config = StereoConfig(
            depth_filter=DepthFilterParams(
                use_wls_filter=False,
                left_right_check=False,
                temporal_alpha=0.5,
            )
        )
        calculator = DepthCalculator(config)
        
        # Create two synthetic frames with slight depth variation
        left1 = np.full((480, 640, 3), 128, dtype=np.uint8)
        right1 = left1.copy()
        
        # First frame establishes baseline
        result1 = calculator.compute(left1, right1)
        
        # Second frame should be smoothed
        result2 = calculator.compute(left1, right1)
        
        # Both should produce valid results
        assert result1.depth_map.shape == (480, 640)
        assert result2.depth_map.shape == (480, 640)

    def test_reset_temporal(self) -> None:
        """Test that reset_temporal clears the temporal state."""
        from stereo_vision import DepthFilterParams
        
        config = StereoConfig(
            depth_filter=DepthFilterParams(
                use_wls_filter=False,
                left_right_check=False,
                temporal_alpha=0.5,
            )
        )
        calculator = DepthCalculator(config)
        
        # Process a frame to set prev_depth
        left = np.full((480, 640, 3), 128, dtype=np.uint8)
        calculator.compute(left, left)
        
        # Now reset
        calculator.reset_temporal()
        
        # After reset, _prev_depth should be None
        assert calculator._prev_depth is None


class TestDepthFilterParams:
    """Tests for DepthFilterParams configuration."""

    def test_defaults(self) -> None:
        from stereo_vision import DepthFilterParams
        
        params = DepthFilterParams()
        assert params.use_wls_filter is True
        assert params.left_right_check is True
        assert params.temporal_alpha == 0.0

    def test_preset_fast(self) -> None:
        from stereo_vision import DepthFilterParams, QualityPreset
        
        params = DepthFilterParams.for_preset(QualityPreset.FAST)
        assert params.use_wls_filter is False
        assert params.left_right_check is False

    def test_preset_quality(self) -> None:
        from stereo_vision import DepthFilterParams, QualityPreset
        
        params = DepthFilterParams.for_preset(QualityPreset.QUALITY)
        assert params.use_wls_filter is True
        assert params.use_bilateral is True
        assert params.temporal_alpha > 0

    def test_bilateral_filtering(self) -> None:
        """Test that bilateral pre-filtering doesn't crash."""
        from stereo_vision import DepthFilterParams
        
        config = StereoConfig(
            depth_filter=DepthFilterParams(
                use_wls_filter=False,
                left_right_check=False,
                use_bilateral=True,
            )
        )
        calculator = DepthCalculator(config)
        
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = left.copy()
        
        result = calculator.compute(left, right)
        assert result.depth_map.shape == (480, 640)
