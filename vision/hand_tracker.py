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

Two MediaPipe APIs can do the detecting (check V5, design doc): the legacy
`solutions.hands` on the CPU, or the Tasks API HandLandmarker, which runs on
the GPU in 4 ms where the legacy one takes 30. They label hands the opposite
way round, so each carries its own label map.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation

from vision.camera import CameraCalibration
from vision.filters import OneEuroFilter
from vision.handedness import MEDIAPIPE_LABEL_TO_SIDE, MIN_SCORE, TASKS_LABEL_TO_SIDE, Detection, assign_sides
from vision.wrist_pose import WristPose, solve_wrist_pose

SIDES = ("left", "right")

#: The Tasks API's model bundle (hand detector + landmarks), fetched as the
#: README says; data/ is git-ignored.
TASKS_MODEL = Path(__file__).resolve().parents[1] / "data" / "models" / "hand_landmarker.task"

#: On a laptop whose display runs on the integrated GPU, EGL picks that one,
#: which is five times *slower* than the CPU here (140 ms a frame). These send
#: MediaPipe's GL context to the NVIDIA GPU; they only need setting before the
#: landmarker is created. Left alone if the NVIDIA EGL vendor file is absent.
NVIDIA_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
NVIDIA_ENV = {
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "__EGL_VENDOR_LIBRARY_FILENAMES": NVIDIA_EGL,
}

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


@dataclass(frozen=True)
class RawHand:
    image_landmarks: np.ndarray       # (21, 3) x, y over image width, height
    world_landmarks: np.ndarray       # (21, 3) metres, hand-centred
    label: str                        # the API's handedness label
    score: float


def _points(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks])


class _LegacyHands:
    label_to_side = MEDIAPIPE_LABEL_TO_SIDE

    def __init__(self, model_complexity: int):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, rgb: np.ndarray, t: float) -> list[RawHand]:
        res = self._hands.process(rgb)
        return [RawHand(_points(im.landmark), _points(w.landmark), c.classification[0].label, c.classification[0].score)
                for im, w, c in zip(res.multi_hand_landmarks or [], res.multi_hand_world_landmarks or [],
                                    res.multi_handedness or [])]

    def close(self) -> None:
        self._hands.close()


class _TasksHands:
    label_to_side = TASKS_LABEL_TO_SIDE

    def __init__(self, model: Path, gpu: bool):
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        if not model.exists():
            raise FileNotFoundError(f"{model}: fetch it as the README says")
        if gpu and os.path.exists(NVIDIA_EGL):
            for key, value in NVIDIA_ENV.items():
                os.environ.setdefault(key, value)
        delegate = mpt.BaseOptions.Delegate.GPU if gpu else mpt.BaseOptions.Delegate.CPU
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model), delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self._last_ms = -1

    def __call__(self, rgb: np.ndarray, t: float) -> list[RawHand]:
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode wants strictly increasing timestamps
        self._last_ms = ms
        res = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        return [RawHand(_points(im), _points(w), c[0].category_name, c[0].score)
                for im, w, c in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness)]

    def close(self) -> None:
        self._landmarker.close()


class HandTracker:
    """Which detector, from check V5 (design doc), on this laptop:

    * default: legacy API, full model, CPU. 30 ms a frame for two hands, on the
      edge of 30 Hz.
    * `model_complexity=0`: legacy lite model. 15 ms, about a quarter more
      jitter, clearly worse near the image edges.
    * `gpu=True`: Tasks API on the GPU. 4 ms, jitter as low as the default.
      Needs `TASKS_MODEL`.
    """

    def __init__(self, calibration: Optional[CameraCalibration] = None, min_score: float = MIN_SCORE,
                 model_complexity: int = 1, gpu: bool = False, tasks_model: Path = TASKS_MODEL):
        self.calibration = calibration
        self.min_score = min_score
        self._detect = _TasksHands(tasks_model, gpu=True) if gpu else _LegacyHands(model_complexity)
        self._filters = {side: _PoseFilter() for side in SIDES}

    @property
    def label_to_side(self) -> dict[str, str]:
        return self._detect.label_to_side

    def detect(self, frame_bgr: np.ndarray, t: float) -> list[RawHand]:
        """Every hand MediaPipe found, before sides are assigned or anything filtered."""
        return self._detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), t)

    def process(self, frame_bgr: np.ndarray, t: float) -> dict[str, HandResult]:
        """Hands found in an unmirrored BGR frame captured at time `t` (seconds)."""
        h, w = frame_bgr.shape[:2]
        if self.calibration is not None and (w, h) != self.calibration.image_size:
            raise ValueError(f"frame is {w}x{h}, the intrinsics are for "
                             f"{self.calibration.image_size[0]}x{self.calibration.image_size[1]}")

        raw = self.detect(frame_bgr, t)
        dets = [Detection(r.label, r.score, r.image_landmarks[0, 0]) for r in raw]
        out = {}
        for side, i in assign_sides(dets, self.min_score, self.label_to_side).items():
            wrist = None
            if self.calibration is not None:
                pose = solve_wrist_pose(raw[i].world_landmarks, raw[i].image_landmarks, (w, h),
                                        self.calibration.camera_matrix, side, self.calibration.dist_coeffs)
                if pose is not None:
                    wrist = self._filters[side](pose, t)
            out[side] = HandResult(world_landmarks=raw[i].world_landmarks, wrist=wrist)
        return out

    def close(self) -> None:
        self._detect.close()
