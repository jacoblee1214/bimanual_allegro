#!/usr/bin/env python3
"""
extract_ego_detections.py — run the detector once over a tools/record_ego_depth.py clip.

So that every offline study of that clip (depth fusion, hand-scale calibration) starts
from the same detections, sides and PnP poses as the live ego tracker
(vision/hand_tracker_ego.py: sides by position, SQPnP with config/ego_camera_info.yaml),
unfiltered.

Writes data/results/m5_detections_<clip>.npz with
  t (N,), phase (N,) str          from the clip's frames.jsonl
  <side>_found (N,) bool          for side in left, right
  <side>_image (N, 21, 3)         MediaPipe image landmarks (x, y over width, height)
  <side>_world (N, 21, 3)         MediaPipe world landmarks, metres
  <side>_score (N,)
  <side>_p (N, 3), <side>_R (N, 3, 3), <side>_reproj (N,)   raw PnP wrist pose (NaN if none)
Frame i here is row i of the clip's depth.npy.

Usage (with .venv-vision):
    tools/extract_ego_detections.py data/clips/ego_depth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision import camera  # noqa: E402
from vision.hand_tracker import HandTracker  # noqa: E402
from vision.handedness import Detection  # noqa: E402
from vision.handedness_ego import EgoSides  # noqa: E402
from vision.wrist_pose import solve_wrist_pose  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", type=Path)
    args = p.parse_args()

    rows = [json.loads(line) for line in (args.clip / "frames.jsonl").open()]
    n = len(rows)
    cal = camera.load(REPO / "config" / "ego_camera_info.yaml")
    tracker, sides = HandTracker(gpu=True), EgoSides()
    out: dict[str, np.ndarray] = {"t": np.array([r["t"] for r in rows]), "phase": np.array([r["phase"] for r in rows])}
    for s in ("left", "right"):
        out[f"{s}_found"] = np.zeros(n, bool)
        out[f"{s}_score"] = np.full(n, np.nan)
        for key, shape in (("image", (21, 3)), ("world", (21, 3)), ("p", (3,)), ("R", (3, 3)), ("reproj", ())):
            out[f"{s}_{key}"] = np.full((n, *shape), np.nan)

    video = cv2.VideoCapture(str(args.clip / "color.avi"))
    for i, row in enumerate(rows):
        ok, frame = video.read()
        if not ok:
            sys.exit(f"color.avi ends at frame {i}, frames.jsonl has {n}")
        h, w = frame.shape[:2]
        raw = tracker.detect(frame, row["t"])
        dets = [Detection(r.label, r.score, r.image_landmarks[0, 0]) for r in raw]
        for s, k in sides(dets, row["t"]).items():
            out[f"{s}_found"][i] = True
            out[f"{s}_image"][i], out[f"{s}_world"][i], out[f"{s}_score"][i] = raw[k].image_landmarks, raw[k].world_landmarks, raw[k].score
            pose = solve_wrist_pose(raw[k].world_landmarks, raw[k].image_landmarks, (w, h), cal.camera_matrix, s, cal.dist_coeffs)
            if pose is not None:
                out[f"{s}_p"][i], out[f"{s}_R"][i], out[f"{s}_reproj"][i] = pose.position, pose.rotation, pose.reprojection_px
    tracker.close()

    dest = REPO / "data" / "results" / f"m5_detections_{args.clip.name}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"{n} frames; found left {out['left_found'].sum()}, right {out['right_found'].sum()} -> {dest}")


if __name__ == "__main__":
    main()
