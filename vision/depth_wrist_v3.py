"""
depth_wrist_v3.py — v2's sensor palm-plane wrist, with a hold that does not trust PnP's distance.

vision/depth_wrist_v2.py puts the wrist on the sensor's palm plane. Without a usable
sensor value (no palm depth, the hand closer than the D457's minimum range, an edge-on
palm, or a ratio outside the absolute band) it falls back on "hold": PnP's wrist times
the median of the recent ratios sensor distance / PnP distance. That median is only as
good as PnP's distance over the last second. On the live dry run before R6 PnP came
closer for 0.6 s just before a 2 s gap, the ratios went from 0.65 to 1.0-1.4, the held
ratio became 1.02, the wrist sat about 30% too far for the whole gap and jumped 123 mm
when the sensor came back (docs/results/r6_ego_depth_follow.md).

Here "hold" keeps the last sensor wrist's distance from the camera (its range along its
ray) and puts the wrist on the current landmark-0 pixel ray at that range. The lateral
position keeps following the image; the distance stays where the sensor last saw it
and does not use PnP at all. On ego_depth_424 with simulated gaps (docs/results/m7_hold_v3.md)
this was closer to the sensor wrist than v2 during every kind of gap, and a PnP error just
before the gap left it unchanged. Following PnP's relative distance change since the last
sensor frame was worse; a smoothed form of it (tuned on that clip) was better on the
right hand's 1-2 s gaps without PnP errors, but much worse with them. Falling back to PnP
times a ratio after a maximum hold time was worse on 2 s gaps. Neither is used.

Sources, as in v2:
  "depth"  this frame's sensor plane (v2 unchanged), with its palm frame as `rotation`
  "hold"   the last sensor range on this frame's landmark-0 ray, `rotation` None
  "pnp"    raw PnP, before the first sensor frame (or after reset())
`scale` is the wrist distance over PnP's (on "hold" frames: what the hold amounts to),
`measured` this frame's ratio before the band, `pixels` the sensor pixels palm_depth kept
(None without a sensor value), for the node's /<side>/wrist_depth_diag.

NumPy only (no OpenCV), so this also imports next to pinocchio in .venv-arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from vision.depth_filter import N_LANDMARKS, WRIST, palm_frame
from vision.depth_wrist import SCALE_BAND, palm_depth
from vision.depth_wrist_v2 import FusedWristV2, on_plane, pixel_rays, sensor_plane

if TYPE_CHECKING:  # vision.wrist_pose imports OpenCV; only the attributes are used here
    from vision.wrist_pose import WristPose


@dataclass(frozen=True)
class FusedWristV3(FusedWristV2):
    pixels: Optional[int] = None  # sensor pixels palm_depth kept, None without a sensor value


class WristDepthV3:
    """Per-hand wrist from the sensor's palm plane; without it, the last sensor range on the current ray."""

    def __init__(self, band: tuple[float, float] = SCALE_BAND) -> None:
        self.band = band
        self.reset()

    def reset(self) -> None:
        self._range: Optional[float] = None          # last sensor wrist distance from the camera, metres
        self._size: Optional[tuple[int, int]] = None  # (w, h) of the last depth image, for the pixel ray

    @property
    def held(self) -> Optional[float]:
        """The range a "hold" frame would use, metres, or None before the first sensor frame."""
        return self._range

    def measure(
        self, image_landmarks: np.ndarray, depth_m: np.ndarray, camera_matrix: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[int]]:
        """(21, 3) landmarks cut by the sensor's palm plane (or None), and the pixels palm_depth kept (or None)."""
        sensor = palm_depth(depth_m, image_landmarks)
        if sensor is None:
            return None, None
        h, w = depth_m.shape
        image = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
        return on_plane(sensor_plane(sensor, camera_matrix), image[:, :2] * (w, h), camera_matrix), sensor.n

    def __call__(
        self,
        pose: "WristPose",
        image_landmarks: np.ndarray,
        world_landmarks: np.ndarray,
        depth_m: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        side: str,
    ) -> FusedWristV3:
        # world_landmarks: unused, kept so this is called like the tracker calls the rest.
        p = np.asarray(pose.position, dtype=np.float64)
        hand, pixels = (None, None) if depth_m is None else self.measure(image_landmarks, depth_m, camera_matrix)
        if depth_m is not None:
            self._size = depth_m.shape[1], depth_m.shape[0]
        s = None
        if hand is not None:
            s = float(np.linalg.norm(hand[WRIST]) / np.linalg.norm(p))
        if s is not None and self.band[0] <= s <= self.band[1]:
            self._range = float(np.linalg.norm(hand[WRIST]))
            return FusedWristV3(hand[WRIST].copy(), s, "depth", s, palm_frame(hand, side), pixels)
        if self._range is not None:
            wrist = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)[WRIST, :2] * self._size
            ray = pixel_rays(wrist, camera_matrix)[0]
            q = ray * (self._range / np.linalg.norm(ray))
            return FusedWristV3(q, float(self._range / np.linalg.norm(p)), "hold", s, None, pixels)
        return FusedWristV3(p.copy(), 1.0, "pnp", s, None, pixels)
