#!/usr/bin/env python3
"""
bench_tracker.py — design doc check V5: how fast, and at what cost, on recorded clips.

No camera or ROS: replays the V2 clips (data/clips, from
tools/check_label_stability.py) through MediaPipe and the wrist PnP, once per
detector (legacy full and lite models on the CPU, the Tasks API on the GPU if
its model is fetched), and reports

  * time per frame, detection (MediaPipe, colour conversion included) and PnP
    separately, the first five frames left out as warm-up;
  * wrong labels, against the left-right order of uncrossed clips;
  * how often the palm normal points the way the clip says the palm faces;
  * jitter: median second difference of the right wrist and index tip
    between consecutive frames. Real hands move smoothly, so most of it is
    estimation noise; compare models with it, do not read it as an error.

Intrinsics are a nominal guess unless --camera-info is given, so distances and
jitter in mm are only comparable between models, not absolute.

Usage:
    tools/bench_tracker.py                        # all four V2 clips
    tools/bench_tracker.py --clips palm2 back --camera-info config/camera_hp_fhd.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision import camera  # noqa: E402
from vision.hand_tracker import TASKS_MODEL, HandTracker  # noqa: E402
from vision.handedness import Detection, assign_sides  # noqa: E402
from vision.wrist_pose import solve_wrist_pose  # noqa: E402

#: Which way the palms face in each clip: True towards the camera, None unknown.
FACING = {"palm": True, "palm2": True, "back": False, "cross": None}
CROSSED = {"cross"}
NOMINAL_K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def second_diff_mm(runs: list[list[np.ndarray]]) -> float:
    per_run = [np.median(np.linalg.norm(np.diff(np.array(r), n=2, axis=0), axis=1)) for r in runs if len(r) > 10]
    return float(np.median(per_run)) * 1000 if per_run else float("nan")


DETECTORS = {"full": dict(model_complexity=1), "lite": dict(model_complexity=0), "gpu": dict(gpu=True)}


def bench(path: Path, detector: str, k: np.ndarray, dist: np.ndarray, facing, crossed: bool) -> dict:
    cap = cv2.VideoCapture(str(path))
    tracker = HandTracker(**DETECTORS[detector])
    label_to_side = tracker.label_to_side
    t_mp, t_pnp, wrong, labelled, normal_ok, reproj = [], [], 0, 0, [], []
    wrist_runs, tip_runs, wrist_run, tip_run = [], [], [], []
    frames = two = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        raw = tracker.detect(frame, (frames - 1) / 30.0)
        t1 = time.perf_counter()
        dets = [Detection(r.label, r.score, r.image_landmarks[0, 0]) for r in raw]
        if len(dets) == 2:
            two += 1
            if not crossed:
                for truth, d in zip(("right", "left"), sorted(dets, key=lambda d: d.wrist_x)):
                    wrong += label_to_side[d.label] != truth
                    labelled += 1
        right = None
        for side, i in assign_sides(dets, label_to_side=label_to_side).items():
            wpts, ipts = raw[i].world_landmarks, raw[i].image_landmarks
            pose = solve_wrist_pose(wpts, ipts, (w, h), k, side, dist)
            if pose is None:
                continue
            reproj.append(pose.reprojection_px)
            if facing is not None:
                normal_ok.append((pose.rotation[2, 0] < 0) == facing)
            if side == "right":
                right = (pose.position, wpts[8])
        t_pnp.append(time.perf_counter() - t1)
        t_mp.append(t1 - t0)
        if right is not None:
            wrist_run.append(right[0])
            tip_run.append(right[1])
        else:
            wrist_runs.append(wrist_run)
            tip_runs.append(tip_run)
            wrist_run, tip_run = [], []
    wrist_runs.append(wrist_run)
    tip_runs.append(tip_run)
    tracker.close()
    t_mp, t_pnp = t_mp[5:], t_pnp[5:]
    total = np.array(t_mp) + np.array(t_pnp)
    return {
        "frames": frames, "two": two,
        "mp_ms": np.median(t_mp) * 1000, "pnp_ms": np.median(t_pnp) * 1000,
        "fps": 1.0 / np.median(total), "p95_ms": np.percentile(total, 95) * 1000,
        "wrong": None if crossed else f"{wrong}/{labelled}",
        "normal": None if not normal_ok else np.mean(normal_ok),
        "reproj": np.median(reproj) if reproj else float("nan"),
        "wrist_jitter": second_diff_mm(wrist_runs), "tip_jitter": second_diff_mm(tip_runs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", default=list(FACING), help="V2 clip names under data/clips")
    p.add_argument("--camera-info", type=Path, help="intrinsics YAML; a nominal guess otherwise")
    args = p.parse_args()

    if args.camera_info:
        cal = camera.load(args.camera_info)
        k, dist = cal.camera_matrix, cal.dist_coeffs
    else:
        k, dist = NOMINAL_K, np.zeros(5)
        print("[i] nominal intrinsics: mm figures compare models, they are not absolute")

    print(f"{'clip':6s} {'model':5s} | 2 hands | MP ms | PnP ms | fps  | p95 ms | wrong labels | palm normal ok | "
          f"reproj px | wrist jitter mm | tip jitter mm")
    for name in args.clips:
        path = REPO / "data" / "clips" / f"v2_{name}.avi"
        if not path.exists():
            print(f"[!] {path} missing, skipped")
            continue
        for model in DETECTORS:
            if model == "gpu" and not TASKS_MODEL.exists():
                continue
            r = bench(path, model, k, dist, FACING.get(name), name in CROSSED)
            normal = "-" if r["normal"] is None else f"{r['normal']:.1%}"
            print(f"{name:6s} {model:5s} | {r['two']:3d}/{r['frames']:3d} | {r['mp_ms']:5.1f} | {r['pnp_ms']:5.2f}  | "
                  f"{r['fps']:4.0f} | {r['p95_ms']:6.1f} | {r['wrong'] or '-':>12s} | {normal:>14s} | "
                  f"{r['reproj']:9.1f} | {r['wrist_jitter']:15.2f} | {r['tip_jitter']:13.2f}")


if __name__ == "__main__":
    main()
