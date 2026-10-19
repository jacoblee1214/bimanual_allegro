"""vision/hand_tracker.py on the V2 clips (data/clips, git-ignored; skipped where absent).

Each check runs on both detectors: the legacy CPU one, and the Tasks API on the
GPU (skipped without its model or a GPU). They label hands the opposite way
round, so the palm test also guards that each uses its own label map.

Intrinsics are a nominal guess (fx = fy = 600 at 640x480), so no distance is
checked here, only what holds under any plausible camera: which hand is where,
and which way the palms face.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.camera import CameraCalibration
from vision.hand_tracker import TASKS_MODEL, HandTracker

CLIPS = Path(__file__).resolve().parents[1] / "data" / "clips"
NOMINAL = CameraCalibration((640, 480), np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]), np.zeros(5))
FRAMES = 90


def clip_path(clip):
    path = CLIPS / f"v2_{clip}.avi"
    if not path.exists():
        pytest.skip(f"{path} not recorded on this machine (tools/check_label_stability.py)")
    return path


def make_tracker(backend, calibration=NOMINAL):
    if backend == "legacy":
        return HandTracker(calibration)
    if not TASKS_MODEL.exists():
        pytest.skip(f"{TASKS_MODEL} not fetched (README)")
    try:
        return HandTracker(calibration, gpu=True)
    except RuntimeError as exc:  # no usable GPU on this machine
        pytest.skip(f"GPU delegate unavailable: {str(exc)[:80]}")


def track(clip, backend="legacy", calibration=NOMINAL):
    cap = cv2.VideoCapture(str(clip_path(clip)))
    tracker = make_tracker(backend, calibration)
    out = []
    for i in range(FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        out.append(tracker.process(frame, i / 30.0))
    tracker.close()
    return out


BACKENDS = ["legacy", "gpu"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("clip, towards_camera", [("palm2", True), ("back", False)])
def test_palm_normal_follows_which_side_faces_the_camera(clip, towards_camera, backend):
    """Both hands, so this also checks the left hand's flipped helper axis."""
    frames = track(clip, backend)
    for side in ("left", "right"):
        normals_z = [f[side].wrist.rotation[2, 0] for f in frames if side in f and f[side].wrist]
        assert len(normals_z) >= 0.9 * len(frames)
        facing = np.mean(np.array(normals_z) < 0)  # camera z points away from the camera
        assert facing >= 0.95 if towards_camera else facing <= 0.05


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("clip", ["palm2", "back"])
def test_uncrossed_right_hand_is_on_the_cameras_left(clip, backend):
    both = [f for f in track(clip, backend) if "left" in f and "right" in f]
    assert len(both) >= 0.9 * FRAMES
    assert all(f["right"].wrist.position[0] < f["left"].wrist.position[0] for f in both)
    assert all(np.isclose(np.linalg.det(f[s].wrist.rotation), 1.0) for f in both for s in ("left", "right"))


def test_frame_size_must_match_the_intrinsics():
    tracker = HandTracker(NOMINAL)
    with pytest.raises(ValueError, match="1280x720"):
        tracker.process(np.zeros((720, 1280, 3), np.uint8), 0.0)
    tracker.close()


def test_without_intrinsics_only_landmarks_come_out():
    hands = [r for f in track("palm2", calibration=None) for r in f.values()]
    assert hands and all(r.wrist is None and r.world_landmarks.shape == (21, 3) for r in hands)
