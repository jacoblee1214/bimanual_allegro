"""
depth_wrist_v2.py — the wrist and the palm frame read off the sensor's palm surface, not PnP's rotation.

vision/depth_wrist.py (M5) scales the PnP hand so that its palm sits at the sensor's palm
depth. That fixes the palm's distance, but the wrist's depth relative to the palm still
comes from PnP's rotation. On a nearly flat palm seen obliquely PnP has two near-equal
tilt solutions, and on the left hand of ego_depth_424 it flipped between them in the
'still' phase, taking the wrist 3-4 cm along the ray with it
(docs/results/m5_depth_wrist.md, 5.2; docs/results/m6_wrist_flip.md).

Here the sensor gives the geometry directly, and PnP only stands in when it cannot:

* `sensor_plane`: a total-least-squares plane through the 3D points of the pixels
  `palm_depth` keeps (the shrunk palm polygon, MAD-banded). On the back of the hand
  this is the surface over the metacarpals, which stays flat when the fingers curl.

* The wrist is landmark 0's pixel ray cut by that plane, and the palm frame is
  `palm_frame` of the landmarks cut by the plane (it uses 0, 5, 9 and 13, all palm
  points), so its x axis is the plane normal and its z axis runs from the wrist to the
  middle MCP on the plane. Neither depends on PnP's rotation or on MediaPipe's idea of
  the hand's size.

The plane is the sensor's surface, not the joint centres: the wrist sits about half a
hand's thickness towards the camera, as it did in v1. Not corrected (not measured).

The gate is only v1's absolute band, on the ratio sensor wrist distance / PnP wrist
distance (it keeps a polygon on the table or the background out). v1's relative band
(REL_BAND of the recent median) is not used: the PnP side of the ratio is the part that
flips, so on ego_depth_424 the band turned good sensor frames into PnP-based "hold"
frames and brought the jumps back (left 'still' ray step p95 2.03 mm without it, 78.14
mm with it; docs/results/m6_wrist_flip.md). When the sensor or the band says no, the
wrist is PnP times the median of the recent in-band ratios ("hold"), or raw PnP before
the first ratio ("pnp"), as in v1. `rotation` is the sensor's palm frame on "depth"
frames and None otherwise (the caller has PnP's in the pose).

NumPy only (no OpenCV), so this also imports next to pinocchio in .venv-arm.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from vision.depth_filter import N_LANDMARKS, WRIST, palm_frame
from vision.depth_wrist import (
    HOLD_WINDOW,
    SCALE_BAND,
    FusedWrist,
    PalmDepth,
    palm_depth,
)

if TYPE_CHECKING:  # vision.wrist_pose imports OpenCV; only the attributes are used here
    from vision.wrist_pose import WristPose

#: A ray closer than this to lying in the plane (|cos| of ray and normal, about 84 degrees)
#: cuts it too obliquely to trust; same limit as depth_wrist.depth_scale.
MIN_COS = 0.1


@dataclass(frozen=True)
class SensorPlane:
    point: np.ndarray    # (3,) centroid of the kept pixels in the camera frame, metres
    normal: np.ndarray   # (3,) unit normal, pointing away from the camera (normal @ point > 0)
    rms: float           # RMS distance of the kept points from the plane, metres


@dataclass(frozen=True)
class FusedWristV2(FusedWrist):
    rotation: Optional[np.ndarray] = None  # (3, 3) sensor palm frame, axes as columns; None unless "depth"


def pixel_rays(pixels: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """(n, 3) rays (x/z, y/z, 1) through pixels (n, 2) (u, v)."""
    k = np.asarray(camera_matrix, dtype=np.float64)
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([(px[:, 0] - k[0, 2]) / k[0, 0], (px[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(px))])


def sensor_plane(sensor: PalmDepth, camera_matrix: np.ndarray) -> SensorPlane:
    """Plane through the 3D points of the pixels palm_depth kept (total least squares)."""
    pts = pixel_rays(sensor.pixels, camera_matrix) * sensor.depths[:, None]
    centre = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - centre, full_matrices=False)
    normal = vt[-1] if vt[-1] @ centre > 0 else -vt[-1]
    return SensorPlane(point=centre, normal=normal, rms=float(sv[-1] / np.sqrt(len(pts))))


def on_plane(plane: SensorPlane, pixels: np.ndarray, camera_matrix: np.ndarray) -> Optional[np.ndarray]:
    """(n, 3) points where the pixels' rays cut the plane, or None if any ray is nearly in it."""
    rays = pixel_rays(pixels, camera_matrix)
    along = rays @ plane.normal
    if np.any(np.abs(along) / np.linalg.norm(rays, axis=1) < MIN_COS):
        return None
    return rays * ((plane.point @ plane.normal) / along)[:, None]


class WristDepthV2:
    """Per-hand wrist from the sensor's palm plane, with v1's absolute band and fallback. Call once per detection."""

    def __init__(self, band: tuple[float, float] = SCALE_BAND, window: int = HOLD_WINDOW) -> None:
        self.band = band
        self._recent: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._recent.clear()

    @property
    def held(self) -> Optional[float]:
        """Median of the recent in-band ratios, or None before the first."""
        return float(np.median(self._recent)) if self._recent else None

    def measure(
        self, image_landmarks: np.ndarray, depth_m: np.ndarray, camera_matrix: np.ndarray
    ) -> Optional[np.ndarray]:
        """(21, 3) landmarks cut by the sensor's palm plane, camera frame, or None."""
        sensor = palm_depth(depth_m, image_landmarks)
        if sensor is None:
            return None
        h, w = depth_m.shape
        image = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
        return on_plane(sensor_plane(sensor, camera_matrix), image[:, :2] * (w, h), camera_matrix)

    def __call__(
        self,
        pose: "WristPose",
        image_landmarks: np.ndarray,
        world_landmarks: np.ndarray,
        depth_m: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        side: str,
    ) -> FusedWristV2:
        # world_landmarks: unused, kept so this is called like the tracker calls the rest.
        p = np.asarray(pose.position, dtype=np.float64)
        hand = None if depth_m is None else self.measure(image_landmarks, depth_m, camera_matrix)
        s = None
        if hand is not None:
            s = float(np.linalg.norm(hand[WRIST]) / np.linalg.norm(p))
        if s is not None and self.band[0] <= s <= self.band[1]:
            self._recent.append(s)
            return FusedWristV2(hand[WRIST].copy(), s, "depth", s, palm_frame(hand, side))
        held = self.held
        if held is not None:
            return FusedWristV2(p * held, held, "hold", s)
        return FusedWristV2(p.copy(), 1.0, "pnp", s)
