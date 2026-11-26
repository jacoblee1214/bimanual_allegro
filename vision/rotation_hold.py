"""
rotation_hold.py — keep the last sensor palm rotation while the depth fusion has none.

Depth fusion v2 to v4 give the palm's rotation from the sensor plane only on "depth" frames;
on "hold" and "pnp" frames `rotation` is None and the tracker falls back to PnP's rotation,
which is the one that flips (docs/results/m6_wrist_flip.md). That did not matter while the
arms followed the position only. For following the wrist's rotation (docs/13_wrist_rotation.md)
this wrapper repeats the last sensor rotation for up to `max_frames` frames instead; after
that, or before the first sensor frame, the result is passed on unchanged.

    fusion_factory = hold_rotation(WristDepthV3)      # a factory, as the tracker expects
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np


class HoldRotation:
    def __init__(self, fusion, max_frames: int = 30) -> None:
        self._fusion, self.max_frames = fusion, int(max_frames)
        self._R: Optional[np.ndarray] = None
        self._age = 0

    def reset(self) -> None:
        self._fusion.reset()
        self._R, self._age = None, 0

    def __getattr__(self, name):            # anything else the tracker asks of the fusion
        return getattr(self._fusion, name)

    def __call__(self, *args, **kwargs):
        fused = self._fusion(*args, **kwargs)
        if getattr(fused, "rotation", None) is not None:
            self._R, self._age = fused.rotation, 0
        elif self._R is not None and self._age < self.max_frames and dataclasses.is_dataclass(fused):
            self._age += 1
            fused = dataclasses.replace(fused, rotation=self._R)
        return fused


def hold_rotation(factory: Callable[[], object], max_frames: int = 30) -> Callable[[], HoldRotation]:
    return lambda: HoldRotation(factory(), max_frames)
