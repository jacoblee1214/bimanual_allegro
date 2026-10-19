#!/usr/bin/env python3
"""
extract_wrist_trajectories.py — filtered wrist poses from recorded clips, for offline arm work.

Runs the same HandTracker the vision node uses over each clip and saves, per clip,
data/trajectories/<clip>.npz with, for each side present:
  <side>_t     (N,)    seconds from the clip start
  <side>_p     (N, 3)  wrist position in camera_optical, metres
  <side>_q     (N, 4)  wrist orientation, quaternion x y z w (scipy order)
  <side>_reproj (N,)   PnP reprojection error, pixels
plus `fps` and `fx` (the focal length used).

Without --camera-info the intrinsics are a nominal guess (fx = fy = --fx, centre of the
image, no distortion): distances scale with fx, so these trajectories are for checking
reachability and joint speeds, not absolute positions (docs/03_arm_follow_design.md, 4).

Usage:
    tools/extract_wrist_trajectories.py data/clips/v2_*.avi --gpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import camera  # noqa: E402
from vision.hand_tracker import SIDES, HandTracker  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "trajectories"


def nominal(size: tuple[int, int], fx: float) -> camera.CameraCalibration:
    w, h = size
    K = np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]])
    return camera.CameraCalibration(image_size=(w, h), camera_matrix=K, dist_coeffs=np.zeros(5))


def extract(clip: Path, calibration_path: Path | None, fx: float, gpu: bool) -> Path:
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    calibration = camera.load(calibration_path) if calibration_path else nominal(size, fx)
    tracker = HandTracker(calibration, gpu=gpu)
    rows: dict[str, list] = {s: [] for s in SIDES}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        for side, hand in tracker.process(frame, t).items():
            if hand.wrist is not None:
                q = Rotation.from_matrix(hand.wrist.rotation).as_quat()
                rows[side].append((t, *hand.wrist.position, *q, hand.wrist.reprojection_px))
        i += 1
    tracker.close()
    cap.release()

    out: dict[str, np.ndarray] = {"fps": np.float64(fps), "fx": np.float64(calibration.camera_matrix[0, 0])}
    for side, r in rows.items():
        if not r:
            continue
        a = np.asarray(r, dtype=np.float64)
        out[f"{side}_t"], out[f"{side}_p"], out[f"{side}_q"], out[f"{side}_reproj"] = a[:, 0], a[:, 1:4], a[:, 4:8], a[:, 8]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{clip.stem}.npz"
    np.savez(path, **out)
    print(f"{clip.name}: {i} frames, " + ", ".join(f"{s} {len(rows[s])}" for s in SIDES) + f" -> {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+", type=Path)
    p.add_argument("--camera-info", type=Path, help="intrinsics YAML; default is a nominal guess")
    p.add_argument("--fx", type=float, default=600.0, help="nominal focal length in pixels (default 600)")
    p.add_argument("--gpu", action="store_true", help="MediaPipe on the NVIDIA GPU")
    args = p.parse_args()
    for clip in args.clips:
        extract(clip, args.camera_info, args.fx, args.gpu)


if __name__ == "__main__":
    main()
