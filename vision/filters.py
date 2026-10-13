"""
filters.py — the One Euro filter, for the wrist pose.

Casiez, Roussel and Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems", CHI 2012. A low-pass filter whose
cutoff rises with speed: a hand held still is smoothed hard, a moving one
barely lags. A fixed EMA has to trade one for the other.

Tuning, from the paper: with `beta` at 0, lower `min_cutoff` until a still hand
stops jittering; then raise `beta` until fast moves stop lagging.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Per-component One Euro filter over a vector.

    `min_cutoff` is in Hz; `beta` scales the speed (signal units per second)
    that raises the cutoff above it.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0.0:  # same or older timestamp: nothing new to filter over
            return self._x.copy()

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = np.array([_alpha(c, dt) for c in cutoff])
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x.copy()
