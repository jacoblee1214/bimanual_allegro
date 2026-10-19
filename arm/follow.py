"""
follow.py — relative (clutch) wrist-to-EEF mapping and per-step joint velocity limiter.

Monocular wrist depth scales with MediaPipe's hand-size estimate (docs/01, 3.3),
so only motion relative to an engagement pose is sent to the robot. See
docs/03_arm_follow_design.md, section 2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: Camera optical frame (x right, y down, z forward) to robot base frame
#: (x forward, y left, z up), assuming the operator faces the camera and the
#: robot faces the same way as the operator:
#:   base_x = -cam_z,  base_y = cam_x,  base_z = -cam_y.
R_CB = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


class RelativeFollower:
    """Maps relative wrist motion in the camera frame onto an engaged EEF pose in the base frame."""

    def __init__(
        self,
        R_cb: np.ndarray = R_CB,
        scale: float = 1.0,
        workspace_lo: Optional[np.ndarray] = None,
        workspace_hi: Optional[np.ndarray] = None,
    ) -> None:
        self.R_cb = np.asarray(R_cb, dtype=np.float64).reshape(3, 3)
        self.scale = float(scale)
        self.workspace_lo = None if workspace_lo is None else np.asarray(workspace_lo, dtype=np.float64).reshape(3)
        self.workspace_hi = None if workspace_hi is None else np.asarray(workspace_hi, dtype=np.float64).reshape(3)

        self._wrist_p0: Optional[np.ndarray] = None
        self._wrist_R0: Optional[np.ndarray] = None
        self._eef_p0: Optional[np.ndarray] = None
        self._eef_R0: Optional[np.ndarray] = None

    def engage(
        self,
        wrist_p: np.ndarray,
        wrist_R: np.ndarray,
        eef_p: np.ndarray,
        eef_R: np.ndarray,
    ) -> None:
        """Latch the reference wrist pose (camera frame) and EEF pose (robot base frame)."""
        self._wrist_p0 = np.asarray(wrist_p, dtype=np.float64).reshape(3).copy()
        self._wrist_R0 = np.asarray(wrist_R, dtype=np.float64).reshape(3, 3).copy()
        self._eef_p0 = np.asarray(eef_p, dtype=np.float64).reshape(3).copy()
        self._eef_R0 = np.asarray(eef_R, dtype=np.float64).reshape(3, 3).copy()

    def target(self, wrist_p: np.ndarray, wrist_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Target EEF `(p, R)` in the robot base frame for current wrist pose `(wrist_p, wrist_R)`."""
        if self._wrist_p0 is None or self._wrist_R0 is None or self._eef_p0 is None or self._eef_R0 is None:
            raise RuntimeError("engage() must be called before target()")

        wp = np.asarray(wrist_p, dtype=np.float64).reshape(3)
        wR = np.asarray(wrist_R, dtype=np.float64).reshape(3, 3)

        p = self._eef_p0 + self.scale * (self.R_cb @ (wp - self._wrist_p0))
        if self.workspace_lo is not None or self.workspace_hi is not None:
            p = np.clip(p, self.workspace_lo, self.workspace_hi)
        R = (self.R_cb @ wR @ self._wrist_R0.T @ self.R_cb.T) @ self._eef_R0
        return p, R


def limit_step(
    q_prev: np.ndarray,
    q_new: np.ndarray,
    max_vel: float | np.ndarray,
    dt: float,
) -> np.ndarray:
    """Clamp per-joint change from `q_prev` to `q_new` to `max_vel * dt`."""
    prev = np.asarray(q_prev, dtype=np.float64)
    new = np.asarray(q_new, dtype=np.float64)
    max_dq = np.asarray(max_vel, dtype=np.float64) * float(dt)
    return prev + np.clip(new - prev, -max_dq, max_dq)
