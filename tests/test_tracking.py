"""
Unit tests for ball tracking module.
"""

import numpy as np
import pytest

from stereo_vision import Position3D, Velocity3D, TrackState, BallTracker


class TestPosition3D:
    """Tests for Position3D dataclass."""

    def test_creation(self) -> None:
        pos = Position3D(100.0, 200.0, 1000.0)
        assert pos.x == 100.0
        assert pos.y == 200.0
        assert pos.z == 1000.0

    def test_distance_to(self) -> None:
        p1 = Position3D(0, 0, 0)
        p2 = Position3D(3, 4, 0)
        assert p1.distance_to(p2) == pytest.approx(5.0)

    def test_addition(self) -> None:
        p1 = Position3D(1, 2, 3)
        p2 = Position3D(4, 5, 6)
        result = p1 + p2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_subtraction(self) -> None:
        p1 = Position3D(5, 7, 9)
        p2 = Position3D(1, 2, 3)
        result = p1 - p2
        assert result.x == 4
        assert result.y == 5
        assert result.z == 6

    def test_scaled(self) -> None:
        pos = Position3D(10, 20, 30)
        scaled = pos.scaled(0.5)
        assert scaled.x == 5
        assert scaled.y == 10
        assert scaled.z == 15

    def test_as_array(self) -> None:
        pos = Position3D(1, 2, 3)
        arr = pos.as_array()
        assert arr.shape == (3,)
        assert np.allclose(arr, [1, 2, 3])

    def test_from_array(self) -> None:
        arr = np.array([10, 20, 30])
        pos = Position3D.from_array(arr)
        assert pos.x == 10
        assert pos.y == 20
        assert pos.z == 30


class TestVelocity3D:
    """Tests for Velocity3D dataclass."""

    def test_speed(self) -> None:
        vel = Velocity3D(3, 4, 0)
        assert vel.speed == pytest.approx(5.0)

    def test_as_array(self) -> None:
        vel = Velocity3D(1, 2, 3)
        arr = vel.as_array()
        assert np.allclose(arr, [1, 2, 3])


class TestTrackState:
    """Tests for TrackState dataclass."""

    def test_is_valid(self) -> None:
        pos = Position3D(0, 0, 1000)
        vel = Velocity3D(0, 0, 0)
        
        state = TrackState(position=pos, velocity=vel, missed_frames=0)
        assert state.is_valid(max_missed=5) is True
        
        state = TrackState(position=pos, velocity=vel, missed_frames=10)
        assert state.is_valid(max_missed=5) is False


class TestBallTracker:
    """Tests for BallTracker Kalman filter."""

    @pytest.fixture
    def tracker(self) -> BallTracker:
        return BallTracker()

    def test_not_initialized_by_default(self, tracker: BallTracker) -> None:
        assert tracker.is_tracking is False

    def test_update_initializes(self, tracker: BallTracker) -> None:
        pos = Position3D(100, 200, 1000)
        state = tracker.update(pos)
        
        assert tracker.is_tracking is True
        assert state.position.z == pytest.approx(1000, abs=10)

    def test_multiple_updates(self, tracker: BallTracker) -> None:
        # Simulate ball moving
        positions = [
            Position3D(0, 0, 1000),
            Position3D(10, 5, 990),
            Position3D(20, 12, 980),
            Position3D(30, 21, 970),
        ]
        
        for pos in positions:
            state = tracker.update(pos, dt=0.033)
        
        assert tracker.is_tracking
        assert state.age == 4

    def test_predict_position(self, tracker: BallTracker) -> None:
        # Initialize at known position with no motion
        tracker.update(Position3D(0, 0, 1000))
        
        # Predict 0.1 seconds ahead (should include gravity)
        future = tracker.predict_position(0.1)
        
        # Y should increase (gravity pulls down)
        assert future.y > 0

    def test_get_trajectory(self, tracker: BallTracker) -> None:
        tracker.update(Position3D(0, 0, 1000))
        tracker.update(Position3D(100, 10, 900), dt=0.1)
        
        trajectory = tracker.get_trajectory(t_end=0.5, steps=5)
        
        assert len(trajectory) == 6  # steps + 1
        assert isinstance(trajectory[0], Position3D)

    def test_empty_trajectory_when_not_initialized(self, tracker: BallTracker) -> None:
        trajectory = tracker.get_trajectory()
        assert trajectory == []

    def test_reset(self, tracker: BallTracker) -> None:
        tracker.update(Position3D(0, 0, 1000))
        assert tracker.is_tracking
        
        tracker.reset()
        assert tracker.is_tracking is False
        assert tracker.history == []

    def test_predict_no_measurement(self, tracker: BallTracker) -> None:
        tracker.update(Position3D(0, 0, 1000))
        
        # Predict without new measurement
        state = tracker.predict_no_measurement(dt=0.033)
        
        assert state.missed_frames == 1
        assert tracker.is_tracking  # Still tracking

    def test_history_recorded(self, tracker: BallTracker) -> None:
        tracker.update(Position3D(0, 0, 1000))
        tracker.update(Position3D(10, 5, 990))
        tracker.update(Position3D(20, 12, 980))
        
        history = tracker.history
        assert len(history) == 3
