"""
hand_tracker.py — one camera frame in, each visible hand's landmarks and wrist pose out.

No ROS here (hand_tracker_node.py wraps it), so it runs on recorded clips in the
tests. docs/01_vision_design.md, 3.1-3.5:

* MediaPipe sees the **unmirrored** frame, so its world landmarks already have
  the hand's true chirality: the hand repo's retargeter takes them with
  `--no-undo-mirror`.
* Sides come from vision/handedness.py.
* The wrist pose needs the camera's intrinsics; without them only the
  landmarks come out.
* The wrist pose is One Euro filtered per side; the landmarks are not, since
  the retargeter smooths its own input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation

from vision.camera import CameraCalibration
from vision.filters import OneEuroFilter
from vision.handedness import MIN_SCORE, Detection, assign_sides
from vision.wrist_pose import WristPose, solve_wrist_pose

SIDES = ("left", "right")

#: A side unseen for longer than this starts its filter afresh, instead of
#: sliding over from where the hand was last seen.
RESET_GAP_S = 0.3

# Initial One Euro settings, to be tuned on still and moving hands (check V4).
POSITION_FILTER = dict(min_cutoff=1.0, beta=10.0)   # metres
ROTATION_FILTER = dict(min_cutoff=1.0, beta=1.0)    # quaternion components


@dataclass(frozen=True)
class HandResult:
    world_landmarks: np.ndarray       # (21, 3) metres, hand-centred, true chirality
    wrist: Optional[WristPose]        # filtered; None without intrinsics or if PnP failed


class _PoseFilter:
    def __init__(self) -> None:
        self._pos = OneEuroFilter(**POSITION_FILTER)
        self._rot = OneEuroFilter(**ROTATION_FILTER)
        self._last_q: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None

    def __call__(self, pose: WristPose, t: float) -> WristPose:
        if self._last_t is not None and t - self._last_t > RESET_GAP_S:
            self._pos.reset()
            self._rot.reset()
            self._last_q = None
        q = Rotation.from_matrix(pose.rotation).as_quat()
        if self._last_q is not None and np.dot(q, self._last_q) < 0.0:
            q = -q  # same rotation; keep to one hemisphere so the filter does not average q and -q
        q = self._rot(q, t)
        q /= np.linalg.norm(q)
        self._last_q, self._last_t = q, t
        return WristPose(
            position=self._pos(pose.position, t),
            rotation=Rotation.from_quat(q).as_matrix(),
            reprojection_px=pose.reprojection_px,
        )


class HandTracker:
    """`model_complexity` 0 is MediaPipe's lite landmark model: twice as fast with
    two hands (15 vs 30 ms a frame on this laptop's CPU), about a quarter more
    jitter, and clearly worse near the image edges (check V5, design doc).
    """

    def __init__(self, calibration: Optional[CameraCalibration] = None, min_score: float = MIN_SCORE,
                 model_complexity: int = 1):
        self.calibration = calibration
        self.min_score = min_score
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._filters = {side: _PoseFilter() for side in SIDES}

    def process(self, frame_bgr: np.ndarray, t: float) -> dict[str, HandResult]:
        """Hands found in an unmirrored BGR frame captured at time `t` (seconds)."""
        h, w = frame_bgr.shape[:2]
        if self.calibration is not None and (w, h) != self.calibration.image_size:
            raise ValueError(f"frame is {w}x{h}, the intrinsics are for "
                             f"{self.calibration.image_size[0]}x{self.calibration.image_size[1]}")

        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = res.multi_hand_landmarks or []
        world = res.multi_hand_world_landmarks or []
        labels = res.multi_handedness or []
        dets = [Detection(c.classification[0].label, c.classification[0].score, lm.landmark[0].x)
                for lm, c in zip(image, labels)]

        out = {}
        for side, i in assign_sides(dets, self.min_score).items():
            world_pts = np.array([[p.x, p.y, p.z] for p in world[i].landmark])
            wrist = None
            if self.calibration is not None:
                image_pts = np.array([[p.x, p.y, p.z] for p in image[i].landmark])
                raw = solve_wrist_pose(world_pts, image_pts, (w, h), self.calibration.camera_matrix,
                                       side, self.calibration.dist_coeffs)
                if raw is not None:
                    wrist = self._filters[side](raw, t)
            out[side] = HandResult(world_landmarks=world_pts, wrist=wrist)
        return out

    def close(self) -> None:
        self._hands.close()
