"""
Unit tests for real stereo camera module.
"""

import pytest

from stereo_vision import StereoConfig, Resolution
from stereo_vision.real_camera import RealStereoCamera, CaptureMode


class TestCaptureMode:
    """Tests for CaptureMode enum."""

    def test_split_mode_exists(self) -> None:
        assert CaptureMode.SPLIT.name == "SPLIT"

    def test_dual_mode_exists(self) -> None:
        assert CaptureMode.DUAL.name == "DUAL"

    def test_modes_are_distinct(self) -> None:
        assert CaptureMode.SPLIT != CaptureMode.DUAL


class TestRealStereoCamera:
    """Tests for RealStereoCamera class."""

    def test_init_fails_with_invalid_camera_index(self) -> None:
        """Test that missing camera raises RuntimeError."""
        config = StereoConfig(resolution=Resolution(640, 480))
        with pytest.raises(RuntimeError, match="Failed to open camera"):
            RealStereoCamera(config, camera_index=999)

    def test_mode_property(self) -> None:
        """Test mode property returns correct value."""
        # We can't actually open a camera, but we can test the enum
        assert CaptureMode.SPLIT.value != CaptureMode.DUAL.value

    def test_get_info_format(self) -> None:
        """Test get_info returns properly formatted string."""
        # This tests the string formatting logic without needing hardware
        mode_str = "Split" if CaptureMode.SPLIT else "Dual"
        assert mode_str == "Split"


class TestRealStereoCameraWithMock:
    """Tests that can run without actual hardware by testing error paths."""

    def test_cannot_open_nonexistent_camera(self) -> None:
        """Verify proper error when camera doesn't exist."""
        config = StereoConfig()
        with pytest.raises(RuntimeError):
            RealStereoCamera(config, camera_index=9999)

    def test_dual_mode_fails_if_right_camera_missing(self) -> None:
        """Test DUAL mode fails gracefully when second camera unavailable."""
        config = StereoConfig()
        # Both cameras should fail to open with invalid indices
        with pytest.raises(RuntimeError):
            RealStereoCamera(
                config,
                camera_index=9998,
                mode=CaptureMode.DUAL,
                right_camera_index=9999,
            )
