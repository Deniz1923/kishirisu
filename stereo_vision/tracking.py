"""
Ball Tracking and Trajectory Prediction
=========================================

Kalman filter-based tracking for 3D ball position estimation
and trajectory prediction.

This module provides:
- Position3D: 3D position in camera coordinates
- TrackState: Current tracking state with velocity
- BallTracker: Kalman filter tracker with trajectory prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections import deque

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    from .detection import Detection


@dataclass(frozen=True, slots=True)
class Position3D:
    """3D position in camera coordinates (millimeters).
    
    Coordinate system:
    - X: right (positive = right of camera)
    - Y: down (positive = below camera center)
    - Z: forward (depth into scene)
    """
    
    x: float
    y: float
    z: float
    
    def distance_to(self, other: Position3D) -> float:
        """Euclidean distance to another position."""
        return float(np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        ))
    
    def __add__(self, other: Position3D) -> Position3D:
        return Position3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: Position3D) -> Position3D:
        return Position3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def scaled(self, factor: float) -> Position3D:
        return Position3D(self.x * factor, self.y * factor, self.z * factor)
    
    def as_array(self) -> npt.NDArray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: npt.NDArray) -> Position3D:
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def __str__(self) -> str:
        return f"({self.x:.0f}, {self.y:.0f}, {self.z:.0f})mm"


@dataclass(frozen=True, slots=True)
class Velocity3D:
    """3D velocity in mm/second."""
    
    vx: float
    vy: float
    vz: float
    
    @property
    def speed(self) -> float:
        """Magnitude of velocity vector."""
        return float(np.sqrt(self.vx**2 + self.vy**2 + self.vz**2))
    
    def as_array(self) -> npt.NDArray:
        return np.array([self.vx, self.vy, self.vz])
    
    @classmethod
    def from_array(cls, arr: npt.NDArray) -> Velocity3D:
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def __str__(self) -> str:
        return f"({self.vx:.0f}, {self.vy:.0f}, {self.vz:.0f})mm/s"


@dataclass(slots=True)
class TrackState:
    """Current state of a tracked ball.
    
    Attributes:
        position: Current 3D position estimate
        velocity: Current 3D velocity estimate
        confidence: Tracking confidence (0.0 to 1.0)
        age: Number of frames since track started
        missed_frames: Consecutive frames without detection
    """
    
    position: Position3D
    velocity: Velocity3D
    confidence: float = 1.0
    age: int = 1
    missed_frames: int = 0
    
    def is_valid(self, max_missed: int = 5) -> bool:
        """Check if track is still valid."""
        return self.missed_frames <= max_missed


@dataclass
class BallTracker:
    """
    Kalman filter-based ball tracking with trajectory prediction.
    
    Uses a 6-state Kalman filter: [x, y, z, vx, vy, vz]
    
    Example:
        >>> tracker = BallTracker()
        >>> state = tracker.update(detection, depth_mm, dt=0.033)
        >>> future_pos = tracker.predict(t_seconds=0.5)
        >>> trajectory = tracker.get_trajectory()
    """
    
    # Kalman filter parameters
    process_noise: float = 100.0  # mm^2 - how much motion varies
    measurement_noise: float = 50.0  # mm^2 - sensor accuracy
    
    # Track management
    max_missed_frames: int = 10
    min_confidence: float = 0.3
    
    # History
    history_length: int = 30
    
    # Internal state
    _state: npt.NDArray = field(init=False, repr=False)  # [x, y, z, vx, vy, vz]
    _covariance: npt.NDArray = field(init=False, repr=False)  # 6x6
    _initialized: bool = field(default=False, init=False)
    _history: deque = field(init=False, repr=False)
    _age: int = field(default=0, init=False)
    _missed: int = field(default=0, init=False)
    
    def __post_init__(self) -> None:
        self._state = np.zeros(6)
        self._covariance = np.eye(6) * 1000.0
        self._history = deque(maxlen=self.history_length)
    
    def update(
        self,
        position: Position3D,
        dt: float = 0.033,  # ~30 FPS
    ) -> TrackState:
        """
        Update tracker with new measurement.
        
        Args:
            position: Measured 3D position
            dt: Time since last update in seconds
            
        Returns:
            Current track state
        """
        measurement = position.as_array()
        
        if not self._initialized:
            # Initialize state with first measurement
            self._state[:3] = measurement
            self._state[3:] = 0  # Zero velocity initially
            self._initialized = True
            self._age = 1
            self._missed = 0
        else:
            # Predict step
            self._predict(dt)
            
            # Update step
            self._update(measurement)
            self._age += 1
            self._missed = 0
        
        # Store in history
        current_pos = Position3D.from_array(self._state[:3])
        self._history.append(current_pos)
        
        return self._get_state()
    
    def predict_no_measurement(self, dt: float = 0.033) -> TrackState:
        """
        Predict next state without a measurement (ball not detected).
        
        Use when detection is lost temporarily.
        """
        if not self._initialized:
            raise ValueError("Tracker not initialized - call update() first")
        
        self._predict(dt)
        self._missed += 1
        self._age += 1
        
        # Increase uncertainty
        self._covariance *= 1.1
        
        return self._get_state()
    
    def predict_position(self, t_seconds: float) -> Position3D:
        """
        Predict ball position at time t in the future.
        
        Uses simple ballistic model: pos + vel * t + 0.5 * g * t^2
        
        Args:
            t_seconds: Time ahead to predict
            
        Returns:
            Predicted 3D position
        """
        if not self._initialized:
            raise ValueError("Tracker not initialized")
        
        pos = self._state[:3].copy()
        vel = self._state[3:].copy()
        
        # Simple physics: gravity affects Y (down)
        gravity_mms2 = 9810.0  # mm/s^2
        
        # Position = pos + vel*t + 0.5*g*t^2
        predicted = pos + vel * t_seconds
        predicted[1] += 0.5 * gravity_mms2 * t_seconds ** 2  # Y is down
        
        return Position3D.from_array(predicted)
    
    def get_trajectory(
        self,
        t_end: float = 1.0,
        steps: int = 20,
    ) -> list[Position3D]:
        """
        Get predicted trajectory as list of positions.
        
        Args:
            t_end: End time in seconds
            steps: Number of trajectory points
            
        Returns:
            List of predicted positions
        """
        if not self._initialized:
            return []
        
        dt = t_end / steps
        return [self.predict_position(i * dt) for i in range(steps + 1)]
    
    def get_landing_time(self, ground_y: float = 0.0) -> float | None:
        """
        Estimate time until ball reaches ground level.
        
        Args:
            ground_y: Y coordinate of ground plane in mm
            
        Returns:
            Time in seconds, or None if ball won't land
        """
        if not self._initialized:
            return None
        
        y0 = self._state[1]
        vy = self._state[4]
        g = 9810.0  # mm/s^2
        
        # Solve quadratic: y0 + vy*t + 0.5*g*t^2 = ground_y
        a = 0.5 * g
        b = vy
        c = y0 - ground_y
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # Return first positive solution
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        elif t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._state = np.zeros(6)
        self._covariance = np.eye(6) * 1000.0
        self._initialized = False
        self._history.clear()
        self._age = 0
        self._missed = 0
    
    @property
    def is_tracking(self) -> bool:
        """Check if actively tracking."""
        return self._initialized and self._missed <= self.max_missed_frames
    
    @property
    def history(self) -> list[Position3D]:
        """Get position history."""
        return list(self._history)
    
    # Internal Kalman filter methods
    
    def _predict(self, dt: float) -> None:
        """Kalman predict step."""
        # State transition matrix (constant velocity model)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Process noise
        Q = np.eye(6) * self.process_noise * dt
        
        # Add gravity to velocity prediction
        self._state[4] += 9810.0 * dt  # gravity in Y
        
        # Predict
        self._state = F @ self._state
        self._covariance = F @ self._covariance @ F.T + Q
    
    def _update(self, measurement: npt.NDArray) -> None:
        """Kalman update step."""
        # Measurement matrix (we only measure position)
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        
        # Measurement noise
        R = np.eye(3) * self.measurement_noise
        
        # Innovation
        y = measurement - H @ self._state
        S = H @ self._covariance @ H.T + R
        
        # Kalman gain
        K = self._covariance @ H.T @ np.linalg.inv(S)
        
        # Update
        self._state = self._state + K @ y
        self._covariance = (np.eye(6) - K @ H) @ self._covariance
    
    def _get_state(self) -> TrackState:
        """Get current track state."""
        # Confidence from covariance trace
        trace = np.trace(self._covariance[:3, :3])
        confidence = max(0.0, min(1.0, 1.0 - trace / 10000.0))
        
        return TrackState(
            position=Position3D.from_array(self._state[:3]),
            velocity=Velocity3D.from_array(self._state[3:]),
            confidence=confidence,
            age=self._age,
            missed_frames=self._missed,
        )
