"""
handedness_ego.py — left/right for the egocentric camera: position decides, not labels.

Measured on the RealSense above and behind the operator (docs/results/m3_ego_labels.md):
MediaPipe's Tasks API labelled the right hand correctly 82% of the time and the left 70%,
with wrong labels scoring up to 0.97, so neither the labels nor their scores can be
trusted. Position can: seen from behind, the operator's right hand is on the right of
the image (wrist x median 0.70 vs 0.17), and with both hands in view they were never
closer than 0.58 of the image width apart. Crossing the hands breaks this, and the
operator is asked not to.

One hand alone goes to whichever side was last seen nearest to it, within `memory_s`;
with nothing recent it goes by which half of the image it is in. That keeps a hand
working near the middle from flipping sides frame to frame.
"""
from __future__ import annotations

from typing import Optional, Sequence

from vision.handedness import Detection

MIDLINE = 0.5


class EgoSides:
    def __init__(self, memory_s: float = 0.5, midline: float = MIDLINE) -> None:
        self.memory_s = memory_s
        self.midline = midline
        self._last: dict[str, tuple[float, float]] = {}    # side -> (t, wrist x)

    def __call__(self, detections: Sequence[Detection], t: float) -> dict[str, int]:
        """{"left": i, "right": j} into `detections` (at most the two widest apart)."""
        if not detections:
            return {}
        if len(detections) >= 2:
            order = sorted(range(len(detections)), key=lambda i: detections[i].wrist_x)
            sides = {"left": order[0], "right": order[-1]}
        else:
            sides = {self._side_of(detections[0].wrist_x, t): 0}
        for side, i in sides.items():
            self._last[side] = (t, detections[i].wrist_x)
        return sides

    def _side_of(self, x: float, t: float) -> str:
        recent = {s: lx for s, (lt, lx) in self._last.items() if t - lt <= self.memory_s}
        if recent:
            return min(recent, key=lambda s: abs(recent[s] - x))
        return "right" if x >= self.midline else "left"

    def reset(self, side: Optional[str] = None) -> None:
        if side is None:
            self._last.clear()
        else:
            self._last.pop(side, None)
