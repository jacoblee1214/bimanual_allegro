"""
hand_tracker_ego.py — HandTracker for the egocentric camera: sides by position (handedness_ego).

Everything else (detection, PnP wrist pose, One Euro filtering) is vision/hand_tracker.py's;
only how the two detections become "left" and "right" changes. See handedness_ego.py
for why.
"""
from __future__ import annotations

import numpy as np

from vision.hand_tracker import HandResult, HandTracker
from vision.handedness import Detection
from vision.handedness_ego import EgoSides
from vision.wrist_pose import solve_wrist_pose


class EgoHandTracker(HandTracker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sides = EgoSides()

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
                    wrist = self._filters[side](pose, t)
            out[side] = HandResult(world_landmarks=raw[i].world_landmarks, wrist=wrist)
        return out
