"""
gate.py — drop wrist samples no hand can produce, and say when to re-clutch.

M1 (docs/results/m1_arm_ik.md) found the vision's wrist jumping up to 5 m/s and 0.5 rad
in one frame near the image edges: left/right swaps and hands re-found after a drop.
Fed to IK, one such sample asks the joints for 18x their speed limit. A wrist cannot
move like that, so the gate compares each sample against the last one it accepted and
throws it away when the implied speed is too high.

Two cases end in re-clutching instead: the hand was gone for longer than
`reacquire_gap`, or the samples kept being rejected for that long. Either way the old
reference no longer describes where the hand is, so the follower should latch the new
wrist pose to the arm's current pose and carry on from there, with no jump.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np


class Verdict(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REENGAGE = "reengage"   # use this sample as the new clutch reference


class WristGate:
    def __init__(self, max_speed: float = 1.5, max_angular_speed: float = 10.0,
                 reacquire_gap: float = 0.3) -> None:
        """Speeds in m/s and rad/s. 1.5 m/s and 10 rad/s sit above the 95th percentile of
        the clips recorded inside the operating area (0.8 m/s, 2.3 rad/s) and well below
        the edge clip's jumps (5 cm and 0.5 rad in 1/30 s, i.e. 1.5 m/s and 15 rad/s)."""
        self.max_speed = float(max_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.reacquire_gap = float(reacquire_gap)
        self._t: Optional[float] = None     # last accepted sample
        self._p: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._t = self._p = self._R = None

    def __call__(self, t: float, p: np.ndarray, R: np.ndarray) -> Verdict:
        p = np.asarray(p, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        if self._t is None or t - self._t > self.reacquire_gap:
            self._t, self._p, self._R = t, p, R
            return Verdict.REENGAGE

        dt = max(t - self._t, 1e-6)
        speed = np.linalg.norm(p - self._p) / dt
        cos = np.clip((np.trace(self._R.T @ R) - 1.0) / 2.0, -1.0, 1.0)
        angular_speed = np.arccos(cos) / dt
        if speed > self.max_speed or angular_speed > self.max_angular_speed:
            return Verdict.REJECT
        self._t, self._p, self._R = t, p, R
        return Verdict.ACCEPT
