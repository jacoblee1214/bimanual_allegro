"""
depth_wrist_v4.py — v3's hold, but the held range follows the hand's size in the image.

vision/depth_wrist_v3.py keeps the last sensor range on the current landmark-0 ray, so a
hand that moves towards or away from the camera during a gap is only caught when the
sensor comes back (docs/results/m7_hold_v3.md, weakness 1: the step at the end of a 2 s
gap in the recorder's 'depth' phase, p50 left 90.1 / right 97.1 mm). Following PnP's
distance instead is worse, because a PnP error just before the gap contaminates it
(v3_pnprel, same document, 4.3).

Here the distance change comes from the 2D landmarks, not from PnP. `span` is the RMS
distance in pixels of the palm landmarks (0, 1, 5, 9, 13, 17) from their centroid. A hand
twice as far is half as large in the image, so with the span at the last sensor frame as
anchor the held range is

    range = last sensor range * (anchor span / span now) ** gain

limited to [1 / max_rel, max_rel] times the anchor range and moved by at most `step` (a
fraction, in log range) per frame towards that value. Both the anchor span and the
current span are medians over the last SPAN_FRAMES frames. The span also changes when the
hand turns, so `gain` < 1 (0.5 by default: on ego_depth_424 gains 0.3-0.7 were all better than v3 and 1.0 overshoots) damps it and the limits keep a turn from carrying the wrist
away. PnP is not used for the range. Optionally (`pnp_weight` > 0) PnP's own relative
distance change since the anchor is mixed in, but only while it agrees with the span's
change (|log ratio| < `pnp_tol`); where it disagrees (a contaminated PnP) it is ignored.
With gain 0 this is v3 exactly.

Sensor frames are v3's (and v2's) unchanged and reset the anchor. Sources "depth",
"hold", "pnp" as in v3; `scale` is the wrist distance over PnP's, `gap_factor` the held
range over the anchor range on "hold" frames (None otherwise).

NumPy only (no OpenCV), so this also imports next to pinocchio in .venv-arm.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from vision.depth_filter import N_LANDMARKS, WRIST
from vision.depth_wrist import SCALE_BAND
from vision.depth_wrist_v2 import pixel_rays
from vision.depth_wrist_v3 import FusedWristV3, WristDepthV3

if TYPE_CHECKING:  # vision.wrist_pose imports OpenCV; only the attributes are used here
    from vision.wrist_pose import WristPose

#: Landmarks whose spread is the hand's size: wrist, thumb CMC and the four MCP joints' base.
SPAN_LANDMARKS = (0, 1, 5, 9, 13, 17)
SPAN_FRAMES = 3
#: Defaults. Chosen on ego_depth_424 (tools/eval_depth_wrist_v4.py), not independent of it.
GAIN = 0.5
MAX_REL = 1.5
STEP = 0.03
PNP_TOL = 0.15


@dataclass(frozen=True)
class FusedWristV4(FusedWristV3):
    gap_factor: Optional[float] = None  # held range / anchor range on "hold" frames


def hand_span(image_landmarks: np.ndarray, size: tuple[int, int]) -> float:
    """RMS distance in pixels of SPAN_LANDMARKS from their centroid (size = (w, h) of the image)."""
    pts = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)[list(SPAN_LANDMARKS), :2] * size
    return float(np.sqrt(np.mean(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1))))


class WristDepthV4(WristDepthV3):
    """v3's wrist; during a gap the held range follows the hand's apparent size."""

    def __init__(
        self,
        band: tuple[float, float] = SCALE_BAND,
        gain: float = GAIN,
        max_rel: float = MAX_REL,
        step: float = STEP,
        pnp_weight: float = 0.0,
        pnp_tol: float = PNP_TOL,
    ) -> None:
        self.gain, self.max_rel, self.step = gain, max_rel, step
        self.pnp_weight, self.pnp_tol = pnp_weight, pnp_tol
        super().__init__(band)

    def reset(self) -> None:
        super().reset()
        self._spans: deque[float] = deque(maxlen=SPAN_FRAMES)  # recent spans, every frame
        self._pnp: deque[float] = deque(maxlen=SPAN_FRAMES)    # recent PnP wrist distances
        self._anchor: Optional[tuple[float, float]] = None     # (span, PnP distance) at the last sensor frame
        self._log_f = 0.0                                      # log(held range / anchor range)

    def __call__(
        self,
        pose: "WristPose",
        image_landmarks: np.ndarray,
        world_landmarks: np.ndarray,
        depth_m: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        side: str,
    ) -> FusedWristV4:
        p = np.asarray(pose.position, dtype=np.float64)
        d_pnp = float(np.linalg.norm(p))
        f = super().__call__(pose, image_landmarks, world_landmarks, depth_m, camera_matrix, side)
        if self._size is not None:  # span in pixels of the (last known) depth image: right aspect ratio
            self._spans.append(hand_span(image_landmarks, self._size))
        self._pnp.append(d_pnp)
        if f.source == "depth":
            self._anchor = (float(np.median(self._spans)), float(np.median(self._pnp)))
            self._log_f = 0.0
            return FusedWristV4(**vars(f))
        if f.source != "hold" or self._anchor is None or self._spans[-1] <= 0.0:
            return FusedWristV4(**vars(f))
        anchor_span, anchor_pnp = self._anchor
        target = self.gain * np.log(anchor_span / float(np.median(self._spans)))
        if self.pnp_weight > 0.0:
            rel = np.log(float(np.median(self._pnp)) / anchor_pnp)
            if abs(rel - target) < self.pnp_tol:
                target = (1.0 - self.pnp_weight) * target + self.pnp_weight * rel
        lim = np.log(self.max_rel)
        target = float(np.clip(target, -lim, lim))
        self._log_f += float(np.clip(target - self._log_f, -self.step, self.step))
        factor = float(np.exp(self._log_f))
        wrist = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)[WRIST, :2] * self._size
        ray = pixel_rays(wrist, camera_matrix)[0]
        r = self._range * factor
        q = ray * (r / np.linalg.norm(ray))
        return FusedWristV4(q, float(r / d_pnp), "hold", f.measured, None, f.pixels, factor)
