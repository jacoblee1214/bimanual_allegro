"""
hand_tracker_ego_v3.py — the egocentric tracker with the wrist distance taken from the depth sensor.

v2 (hand_tracker_ego_v2.py) filters the PnP wrist along the camera ray with RayKalman,
because PnP's distance follows MediaPipe's wobbling hand-size guess. M5
(docs/results/m5_depth_wrist.md) corrected that distance offline with the RealSense's
own depth of the palm and filtered the result with a RayKalman that trusts the ray
direction on the frames the sensor corrected (sigma_ray 5 mm) and keeps M4's 45 mm on
the others. This tracker does the same per frame, live:

    raw PnP pose -> fusion (sensor depth, aligned to the colour frame) -> RayKalman
    position, with sigma_ray by the fusion's `source`; rotation through One Euro as v2

The fusion is pluggable. `fusion_factory()` is called once per side and must return an
object with

    __call__(pose: WristPose, image_landmarks, world_landmarks, depth_m: Optional[np.ndarray],
             camera_matrix, side) -> FusedWrist
    reset()

The returned FusedWrist (vision/depth_wrist.py) may carry two extra attributes:
`rotation` ((3, 3), used instead of PnP's when present and not None) and `pixels`
(sensor pixels used, for the diagnostics). `DepthFusionV1` wraps M5's palm_depth +
DepthFusion this way and is the default. A side unseen for longer than RESET_GAP_S
has its fusion reset, as its filters are.

Without depth (`depth_m=None`) every frame is a "hold" or "pnp" frame at the M4
sigma_ray, which with DepthFusionV1 is v2's behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from vision.depth_filter import DEFAULT_SIGMA_RAY
from vision.depth_filter_v2 import RayKalmanV2
from vision.depth_wrist import DepthFusion, FusedWrist, palm_depth
from vision.hand_tracker import RESET_GAP_S, SIDES, HandResult, _PoseFilter
from vision.hand_tracker_ego import EgoHandTracker
from vision.handedness import MIN_SCORE, Detection
from vision.handedness_ego import EgoSides
from vision.wrist_pose import WristPose, solve_wrist_pose

#: RayKalman ray noise on frames whose distance came from this frame's sensor depth,
#: metres: M5's SIGMA_RAY_FUSED (tools/eval_depth_wrist.py), kept as it was.
SIGMA_RAY_SENSOR = 0.005
#: FusedWrist.source values that mean "this frame's sensor depth".
SENSOR_SOURCES = ("depth",)


@dataclass(frozen=True)
class SensorFused(FusedWrist):
    pixels: Optional[int] = None   # sensor pixels palm_depth kept, None without a sensor value


class DepthFusionV1:
    """M5's palm_depth + DepthFusion behind the fusion interface above."""

    def __init__(self) -> None:
        self._fusion = DepthFusion()

    def reset(self) -> None:
        self._fusion.reset()

    def __call__(self, pose: WristPose, image_landmarks: np.ndarray, world_landmarks: np.ndarray,
                 depth_m: Optional[np.ndarray], camera_matrix: np.ndarray, side: str) -> SensorFused:
        sensor = palm_depth(depth_m, image_landmarks) if depth_m is not None else None
        f = self._fusion(pose.position, pose.rotation, world_landmarks, sensor, camera_matrix, side)
        return SensorFused(f.position, f.scale, f.source, f.measured, None if sensor is None else sensor.n)


@dataclass(frozen=True)
class DepthHandResult(HandResult):
    fused: Optional[FusedWrist] = None    # the fusion's output, before RayKalman
    sigma_ray: Optional[float] = None     # RayKalman ray noise used for it, metres


class DepthEgoHandTracker(EgoHandTracker):
    def __init__(self, calibration=None, fusion_factory: Callable[[], object] = DepthFusionV1,
                 sigma_ray_sensor: float = SIGMA_RAY_SENSOR, sigma_ray_other: float = DEFAULT_SIGMA_RAY,
                 detector=None, **kwargs) -> None:
        """`detector`: a stand-in for MediaPipe (recorded detections, tests), called as
        detector(rgb, t) -> list[RawHand] and closed with close(); None builds MediaPipe as
        HandTracker does, from `kwargs` (min_score, model_complexity, gpu, tasks_model)."""
        if detector is None:
            super().__init__(calibration, **kwargs)
        else:
            self.calibration = calibration
            self.min_score = kwargs.get("min_score", MIN_SCORE)
            self._detect = detector
            self._filters = {side: _PoseFilter() for side in SIDES}
            self.sides = EgoSides()
        self.sigma_ray_sensor = float(sigma_ray_sensor)
        self.sigma_ray_other = float(sigma_ray_other)
        self._fusion = {side: fusion_factory() for side in SIDES}
        self._ray = {side: RayKalmanV2() for side in SIDES}
        self._last_fused_t: dict[str, float] = {}

    def process(self, frame_bgr: np.ndarray, t: float,
                depth_m: Optional[np.ndarray] = None) -> dict[str, DepthHandResult]:
        """Hands in an unmirrored BGR frame at time `t`; `depth_m`: depth aligned to it, metres, 0 = none."""
        h, w = frame_bgr.shape[:2]
        if self.calibration is not None and (w, h) != self.calibration.image_size:
            raise ValueError(f"frame is {w}x{h}, the intrinsics are for "
                             f"{self.calibration.image_size[0]}x{self.calibration.image_size[1]}")
        if depth_m is not None and depth_m.shape[:2] != (h, w):
            raise ValueError(f"depth is {depth_m.shape[1]}x{depth_m.shape[0]}, the frame {w}x{h}: align it first")
        raw = self.detect(frame_bgr, t)
        dets = [Detection(r.label, r.score, r.image_landmarks[0, 0]) for r in raw]
        out = {}
        for side, i in self.sides(dets, t).items():
            wrist, fused, sigma = None, None, None
            if self.calibration is not None:
                k = self.calibration.camera_matrix
                pose = solve_wrist_pose(raw[i].world_landmarks, raw[i].image_landmarks, (w, h), k, side,
                                        self.calibration.dist_coeffs)
                if pose is not None:
                    last = self._last_fused_t.get(side)
                    if last is not None and t - last > RESET_GAP_S:
                        self._fusion[side].reset()
                    self._last_fused_t[side] = t
                    fused = self._fusion[side](pose, raw[i].image_landmarks, raw[i].world_landmarks, depth_m, k, side)
                    sigma = self.sigma_ray_sensor if fused.source in SENSOR_SOURCES else self.sigma_ray_other
                    rotation = getattr(fused, "rotation", None)
                    rotation = pose.rotation if rotation is None else rotation
                    smoothed = self._filters[side](WristPose(pose.position, rotation, pose.reprojection_px), t)
                    wrist = WristPose(position=self._ray[side](fused.position, t, sigma),
                                      rotation=smoothed.rotation,
                                      reprojection_px=pose.reprojection_px)
            out[side] = DepthHandResult(world_landmarks=raw[i].world_landmarks, wrist=wrist, fused=fused, sigma_ray=sigma)
        return out
