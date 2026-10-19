"""
hand_tracker_ego_v2.py — the egocentric tracker with the wrist position filtered along the camera ray.

v1 (hand_tracker_ego.py) smooths the wrist position with the same One Euro filter in every
direction. But monocular PnP depth, along the camera ray, jitters several times more than
the lateral position, because it scales with MediaPipe's hand-size estimate. Here the raw
PnP position goes through depth_filter.RayKalman instead, which trusts the measurement
across the ray and smooths heavily along it. On the recorded ego clip
(docs/results/m4_depth_filter.md, re-run by the lead), the right wrist's still-hand depth
jitter fell from 10.7 mm (One Euro) to 3.7 mm with 6-13 ms of lateral lag instead of
18-46 ms, and the arm's wrist gate would have rejected 2 samples instead of 75.
Orientation keeps the One Euro filter.
"""
from __future__ import annotations

import numpy as np

from vision.depth_filter import RayKalman
from vision.hand_tracker import HandResult
from vision.hand_tracker_ego import EgoHandTracker
from vision.handedness import Detection
from vision.wrist_pose import WristPose, solve_wrist_pose


class RayEgoHandTracker(EgoHandTracker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ray = {side: RayKalman() for side in self._filters}

    def process(self, frame_bgr: np.ndarray, t: float) -> dict[str, HandResult]:
        h, w = frame_bgr.shape[:2]
        if self.calibration is not None and (w, h) != self.calibration.image_size:
            raise ValueError(f"frame is {w}x{h}, the intrinsics are for "
                             f"{self.calibration.image_size[0]}x{self.calibration.image_size[1]}")
        raw = self.detect(frame_bgr, t)
        dets = [Detection(r.label, r.score, r.image_landmarks[0, 0]) for r in raw]
        out = {}
        for side, i in self.sides(dets, t).items():
            wrist = None
            if self.calibration is not None:
                pose = solve_wrist_pose(raw[i].world_landmarks, raw[i].image_landmarks, (w, h),
                                        self.calibration.camera_matrix, side, self.calibration.dist_coeffs)
                if pose is not None:
                    wrist = WristPose(position=self._ray[side](pose.position, t),
                                      rotation=self._filters[side](pose, t).rotation,
                                      reprojection_px=pose.reprojection_px)
            out[side] = HandResult(world_landmarks=raw[i].world_landmarks, wrist=wrist)
        return out
