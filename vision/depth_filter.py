"""
depth_filter.py — anisotropic depth filtering for monocular PnP wrist tracking.

Monocular PnP places the wrist at a distance proportional to MediaPipe's
per-frame hand size estimate (vision/wrist_pose.py). Frame-to-frame wobble in
that scale turns into wrist position noise directed along the camera ray from
the optical origin to the wrist, 4-5x larger than lateral image-landmark noise
(docs/results/m2_scale_jitter.md, r4_ego_right_wrist_follow.md).

Two causal building blocks address this without touching lateral responsiveness:

* `RayKalman`: a 3D constant-velocity Kalman filter whose measurement
  covariance is a prolate ellipsoid elongated along the unit camera ray `u`
  through the current measurement:
      R = sigma_lat^2 * I + (sigma_ray^2 - sigma_lat^2) * u u^T
  Lateral updates trust the measurement (`sigma_lat` ~ 2 mm) and follow fast
  moves in 1-2 frames; ray updates smooth heavily (`sigma_ray` ~ 45 mm).
  The prior covariance is transported into the current ray basis before each
  update so lateral motion across rays never triangulates into a false depth
  step toward the camera origin.

* `HandSizeHold`: before PnP, rescales MediaPipe's 21 world landmarks so the
  palm size (mean wrist-to-MCP 5, 9, 13, 17 length) equals the causal running
  median over the past `window` detections. In M2, scale wobble drove depth
  jitter when the back of the hand faced the camera, whereas on a palm-facing
  hand MCP length noise was local shape jitter that rescaling injected into
  depth; `back_only=True` applies the hold only when the palm normal from
  `vision/wrist_pose.palm_frame` points away from the camera (+z > 0).
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

N_LANDMARKS = 21
WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP = 0, 5, 9, 13, 17
MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP)

#: Same re-acquisition gap as vision/hand_tracker.py (_PoseFilter).
RESET_GAP_S = 0.3

#: Tuned once on v2_back (M4) and kept fixed across all clips:
#: 2 mm lateral noise, 45 mm ray noise, 0.25 m/s^1.5 acceleration spectral density root.
DEFAULT_SIGMA_LAT = 0.002
DEFAULT_SIGMA_RAY = 0.045
DEFAULT_ACCEL = 0.25
DEFAULT_WINDOW = 45


def palm_frame(points: np.ndarray, side: str) -> np.ndarray:
    """vision.wrist_pose.palm_frame, again in plain NumPy.

    wrist_pose imports OpenCV, and the apt OpenCV that .venv-arm sees is built against
    numpy 1.x, so it cannot be imported next to pinocchio's numpy 2.x. This copy only
    needs the frame's x axis (the palm normal); tests/test_depth_filter.py checks it
    against the original wherever that imports.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    pts = np.asarray(points, dtype=np.float64).reshape(N_LANDMARKS, 3)
    z = pts[MIDDLE_MCP] - pts[WRIST]
    z = z / np.linalg.norm(z)
    y_aux = pts[INDEX_MCP] - pts[RING_MCP]
    if side == "left":
        y_aux = -y_aux
    x = np.cross(y_aux, z)
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def unit_ray(z: np.ndarray) -> np.ndarray:
    """Unit vector from the camera origin through 3D point `z` (defaults to +z at origin)."""
    v = np.asarray(z, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def palm_size(world_landmarks: np.ndarray) -> float:
    """Mean distance from wrist (0) to MCPs (5, 9, 13, 17), in metres."""
    pts = np.asarray(world_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    return float(np.linalg.norm(pts[list(MCPS)] - pts[WRIST], axis=1).mean())


def is_back_facing(world_landmarks: np.ndarray, side: str = "right") -> bool:
    """True when the back of the hand faces the camera (palm normal points along +z, away)."""
    return bool(palm_frame(world_landmarks, side)[2, 0] > 0.0)


class RayKalman:
    """Constant-velocity Kalman filter with ray-elongated measurement covariance.

    State is `[px, py, pz, vx, vy, vz]` in the camera optical frame (metres,
    m/s). Process noise integrates continuous-time white acceleration with
    spectral density `accel^2` over variable `dt`:
        Q(dt) = accel^2 * [[dt^3/3 * I, dt^2/2 * I],
                           [dt^2/2 * I, dt     * I]]
    """

    def __init__(
        self,
        sigma_lat: float = DEFAULT_SIGMA_LAT,
        sigma_ray: float = DEFAULT_SIGMA_RAY,
        accel: float = DEFAULT_ACCEL,
        reset_gap_s: float = RESET_GAP_S,
    ) -> None:
        self.sigma_lat = float(sigma_lat)
        self.sigma_ray = float(sigma_ray)
        self.accel = float(accel)
        self.reset_gap_s = float(reset_gap_s)
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._P_lat: Optional[np.ndarray] = None
        self._P_ray: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def measurement_covariance(self, z: np.ndarray) -> np.ndarray:
        """3x3 measurement covariance R = sigma_lat^2 I + (sigma_ray^2 - sigma_lat^2) u u^T."""
        u = unit_ray(z)
        s_lat2 = self.sigma_lat ** 2
        s_ray2 = self.sigma_ray ** 2
        return s_lat2 * np.eye(3) + (s_ray2 - s_lat2) * np.outer(u, u)

    def __call__(self, z: np.ndarray, t: float) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(3)
        t = float(t)
        if self._x is None or (self._t is not None and t - self._t > self.reset_gap_s):
            self._x = np.r_[z, np.zeros(3)]
            self._P_lat = np.diag([self.sigma_lat ** 2, 0.0])
            self._P_ray = np.diag([self.sigma_ray ** 2, 0.0])
            self._t = t
            return z.copy()

        dt = t - self._t
        if dt <= 0.0:
            return self._x[:3].copy()

        F2 = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.accel ** 2
        Q2 = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
        P_lat_pred = F2 @ self._P_lat @ F2.T + Q2
        P_ray_pred = F2 @ self._P_ray @ F2.T + Q2

        u = unit_ray(z)
        uu = np.outer(u, u)
        P_pred = np.kron(P_lat_pred, np.eye(3) - uu) + np.kron(P_ray_pred, uu)
        R = self.measurement_covariance(z)

        x_pred = np.kron(F2, np.eye(3)) @ self._x
        S = P_pred[:3, :3] + R
        K = np.linalg.solve(S, P_pred[:3, :]).T
        self._x = x_pred + K @ (z - x_pred[:3])

        s_lat2 = self.sigma_lat ** 2
        s_ray2 = self.sigma_ray ** 2
        k_lat = P_lat_pred[:, 0] / (P_lat_pred[0, 0] + s_lat2)
        k_ray = P_ray_pred[:, 0] / (P_ray_pred[0, 0] + s_ray2)
        self._P_lat = (np.eye(2) - np.outer(k_lat, [1.0, 0.0])) @ P_lat_pred
        self._P_ray = (np.eye(2) - np.outer(k_ray, [1.0, 0.0])) @ P_ray_pred
        self._t = t
        return self._x[:3].copy()


class HandSizeHold:
    """Causal pre-PnP world-landmark rescaler holding the palm size to a running median.

    Rescales the 21 world landmarks so the mean wrist-to-MCP (5, 9, 13, 17)
    distance equals the median over the past `window` detections. When
    `back_only=True`, rescaling only applies on frames where the back of the
    hand faces the camera (`palm_frame` normal pointing away from the camera).
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        back_only: bool = False,
        side: str = "right",
        reset_gap_s: float = RESET_GAP_S,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.window = int(window)
        self.back_only = bool(back_only)
        self.side = side
        self.reset_gap_s = float(reset_gap_s)
        self.reset()

    def reset(self) -> None:
        self._sizes: deque[float] = deque(maxlen=self.window)
        self._t: Optional[float] = None

    @property
    def held_size(self) -> Optional[float]:
        """Current running median palm size in metres, or None before the first frame."""
        return float(np.median(self._sizes)) if self._sizes else None

    def __call__(self, world_landmarks: np.ndarray, side: Optional[str] = None,
                 t: Optional[float] = None) -> np.ndarray:
        """`world_landmarks` (21, 3) or (63,) rescaled to the held palm size; same shape out."""
        arr = np.asarray(world_landmarks, dtype=np.float64)
        pts = arr.reshape(N_LANDMARKS, 3)
        side_val = side or self.side
        if t is not None:
            if self._t is not None and t - self._t > self.reset_gap_s:
                self.reset()
            self._t = float(t)

        s = palm_size(pts)
        if s <= 1e-9:
            return arr.copy()
        self._sizes.append(s)

        if self.back_only and not is_back_facing(pts, side_val):
            return arr.copy()

        s_ref = float(np.median(self._sizes))
        scaled = pts * (s_ref / s)
        return scaled.reshape(arr.shape)
