"""
depth_wrist.py — correct the monocular PnP wrist with the RealSense's own depth of the palm.

PnP (vision/wrist_pose.py) gets the hand's direction from the camera right, but its
distance is in proportion to MediaPipe's guess of the hand's size: a hand 10% bigger
than MediaPipe thinks is placed 10% too close. That guess wobbles from frame to
frame and differs between the palm and the back of the hand
(docs/results/m2_scale_jitter.md). The depth sensor measures the distance directly,
so the fix here is one number per frame: the factor that moves the PnP hand along
its rays until its palm sits at the sensor's palm depth.

Two steps:

* `palm_depth`: the sensor's depth inside the palm polygon (landmarks 0, 1, 5, 9,
  13, 17), shrunk towards its centre so that edge pixels stay out. The polygon lies
  inside the hand, so the palm is the majority of its valid pixels; a median and a
  MAD band drop the rest (holes are already 0, plus stray stereo matches). A
  "nearest dense mode" rule was not used: with the hand near the sensor's minimum
  range, the stray matches pile up just there and would be picked first.

* `depth_scale`: the PnP hand is turned into camera-frame points from the pose the
  tracker already has (wrist position and palm frame; the palm frame is
  rotation-equivariant, so it gives back PnP's rotation without solving PnP again).
  A plane through its palm points is cut by the ray of every pixel `palm_depth`
  kept, and the scale is the median of sensor depth over plane depth, pixel by
  pixel. Comparing the same pixels keeps a tilted palm, or a palm with holes on
  one side, from biasing the ratio. Scaling every camera-frame point by that
  factor leaves the image, and so the lateral position and the rotation, as they
  were.

`DepthFusion` adds the gate. A frame's scale is used when it lies in a wide absolute
band and, once a few frames have been seen, within REL_BAND of the median of the
recent in-band scales. The absolute band cannot be tight, because the operator's
hand may differ from MediaPipe's idea of a hand by tens of percent; the relative
band then catches the frames whose polygon sits on something else. The median
is over all recent in-band scales, used or not, so a wrong start cannot lock
the gate shut. Without a usable scale the wrist falls back on PnP times that
median (`hold`), not on raw PnP: most of PnP's scale error is that steady bias,
and dropping back to unscaled PnP would make the wrist jump by it.

NumPy only, so this also imports next to pinocchio in .venv-arm.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vision.depth_filter import N_LANDMARKS, WRIST, palm_frame

#: Wrist, thumb CMC and the four finger MCPs: the part of the hand that stays flat.
PALM = (0, 1, 5, 9, 13, 17)

#: Polygon shrunk to this fraction about its vertex mean, to keep edge pixels out.
SHRINK = 0.8
#: Fewer kept pixels than this, or valid pixels covering less than MIN_FILL of the
#: polygon, and the frame has no sensor value.
MIN_PIXELS = 30
MIN_FILL = 0.3
#: Valid sensor range, metres. The D4xx reports some failed pixels as 65535 counts.
MIN_RANGE_M, MAX_RANGE_M = 0.1, 3.0
#: MAD band half-width (in robust standard deviations) and its floor, metres.
MAD_K = 3.0
MAD_FLOOR_M = 0.005
#: Plausible sensor / PnP scale, before anything is known about this operator.
SCALE_BAND = (0.5, 2.0)
#: Once MIN_HOLD in-band scales are in the window, a frame's scale must also be within
#: this fraction of their median. MediaPipe's palm size wobbles 2-12% (CV, M4) and
#: differs ~9% between palm and back (M2), so 25% leaves room for both.
REL_BAND = 0.25
MIN_HOLD = 5
#: Recent in-band scales whose median gates and stands in for missing frames.
HOLD_WINDOW = 30


@dataclass(frozen=True)
class PalmDepth:
    z: float              # median sensor depth of the kept pixels, metres along the optical axis
    n: int                # kept pixels
    spread: float         # robust standard deviation of the kept depths (1.4826 MAD), metres
    fill: float           # valid pixels / pixels inside the polygon
    pixels: np.ndarray    # (n, 2) kept pixels, (u, v)
    depths: np.ndarray    # (n,) their sensor depths, metres


def polygon_pixels(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(n, 2) integer pixels (u, v) whose centres lie inside `polygon` (k, 2), even-odd rule.

    Pixel (u, v) has its centre at (u, v), as in cv2.projectPoints and so in PnP.
    """
    h, w = shape
    lo = np.maximum(np.ceil(polygon.min(axis=0)), 0).astype(int)
    hi = np.minimum(np.floor(polygon.max(axis=0)), (w - 1, h - 1)).astype(int)
    if np.any(hi < lo):
        return np.empty((0, 2), dtype=int)
    uu, vv = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    inside = np.zeros(uu.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if y1 == y2:
            continue
        crosses = (y1 > vv) != (y2 > vv)
        x_cross = x1 + (vv - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (uu < x_cross)
    return np.column_stack([uu[inside], vv[inside]])


def palm_depth(
    depth_m: np.ndarray,
    image_landmarks: np.ndarray,
    shrink: float = SHRINK,
    min_pixels: int = MIN_PIXELS,
    min_fill: float = MIN_FILL,
) -> Optional[PalmDepth]:
    """Sensor depth of the palm, or None if too few of its pixels have a believable depth.

    Args:
        depth_m: (H, W) depth aligned to the colour image MediaPipe saw, metres, 0 = no data.
        image_landmarks: (21, 3) or (63,) MediaPipe image landmarks (x, y over width, height).
    """
    h, w = depth_m.shape
    image = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    polygon = image[list(PALM), :2] * (w, h)
    centre = polygon.mean(axis=0)
    pixels = polygon_pixels(centre + shrink * (polygon - centre), (h, w))
    if len(pixels) < min_pixels:
        return None

    z = depth_m[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    valid = (z > MIN_RANGE_M) & (z < MAX_RANGE_M)
    if valid.sum() < min_pixels or valid.mean() < min_fill:
        return None

    pixels, z = pixels[valid], z[valid]
    med = np.median(z)
    band = max(MAD_K * 1.4826 * np.median(np.abs(z - med)), MAD_FLOOR_M)
    kept = np.abs(z - med) <= band
    if kept.sum() < min_pixels:
        return None
    z = z[kept]
    med = float(np.median(z))
    return PalmDepth(
        z=med,
        n=int(kept.sum()),
        spread=float(1.4826 * np.median(np.abs(z - med))),
        fill=float(valid.mean()),
        pixels=pixels[kept],
        depths=z,
    )


def hand_in_camera(position: np.ndarray, rotation: np.ndarray, world_landmarks: np.ndarray, side: str) -> np.ndarray:
    """(21, 3) PnP hand in the camera frame, from the wrist pose solve_wrist_pose returned.

    solve_wrist_pose places the world landmarks with a rotation r and returns the palm
    frame of the placed points, which is r @ palm_frame(world). So r comes back as
    rotation @ palm_frame(world).T, and the hand is the world landmarks turned by r
    and moved so that the wrist lands on `position`.
    """
    world = np.asarray(world_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    r = np.asarray(rotation, dtype=np.float64) @ palm_frame(world, side).T
    return np.asarray(position, dtype=np.float64) + (world - world[WRIST]) @ r.T


def depth_scale(sensor: PalmDepth, hand: np.ndarray, camera_matrix: np.ndarray) -> Optional[float]:
    """Median over the kept pixels of sensor depth / depth of the PnP palm plane on that pixel's ray.

    None if the palm is so close to edge-on that the plane depth is meaningless.
    """
    palm = np.asarray(hand, dtype=np.float64)[list(PALM)]
    centre = palm.mean(axis=0)
    normal = np.linalg.svd(palm - centre)[2][-1]
    k = np.asarray(camera_matrix, dtype=np.float64)
    rays = np.column_stack([(sensor.pixels[:, 0] - k[0, 2]) / k[0, 0],
                            (sensor.pixels[:, 1] - k[1, 2]) / k[1, 1],
                            np.ones(len(sensor.pixels))])
    cos = rays @ normal / np.linalg.norm(rays, axis=1)
    ok = np.abs(cos) > 0.1  # within about 84 degrees of facing the camera
    if not ok.any():
        return None
    plane_z = (centre @ normal) / (rays[ok] @ normal)
    return float(np.median(sensor.depths[ok] / plane_z))


@dataclass(frozen=True)
class FusedWrist:
    position: np.ndarray     # (3,) corrected wrist, camera optical frame, metres
    scale: float             # factor applied to the PnP wrist
    source: str              # "depth" (this frame's sensor), "hold" (recent scales) or "pnp" (none yet)
    measured: Optional[float]  # this frame's sensor / PnP scale before the gate, if any


class DepthFusion:
    """Per-hand gate and fallback around `depth_scale`. Call once per detection."""

    def __init__(
        self,
        band: tuple[float, float] = SCALE_BAND,
        rel_band: float = REL_BAND,
        window: int = HOLD_WINDOW,
    ) -> None:
        self.band = band
        self.rel_band = rel_band
        self._recent: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._recent.clear()

    @property
    def held(self) -> Optional[float]:
        """Median of the recent in-band scales, or None before the first."""
        return float(np.median(self._recent)) if self._recent else None

    def __call__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        world_landmarks: np.ndarray,
        sensor: Optional[PalmDepth],
        camera_matrix: np.ndarray,
        side: str,
    ) -> FusedWrist:
        p = np.asarray(position, dtype=np.float64)
        s = None
        if sensor is not None:
            s = depth_scale(sensor, hand_in_camera(p, rotation, world_landmarks, side), camera_matrix)
        held = self.held
        use = s is not None and self.band[0] <= s <= self.band[1]
        if use:
            if len(self._recent) >= MIN_HOLD:
                use = abs(s / held - 1.0) <= self.rel_band
            self._recent.append(s)
        if use:
            return FusedWrist(p * s, s, "depth", s)
        if held is not None:
            return FusedWrist(p * held, held, "hold", s)
        return FusedWrist(p.copy(), 1.0, "pnp", s)
