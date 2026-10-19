#!/usr/bin/env python3
"""
check_labels_ego.py — V1/V2 again, for the egocentric RealSense view (docs/05_ego_camera.md).

From above and behind, the camera sees the backs of the operator's hands, and a right
hand seen from the back looks much like a left hand seen from the palm. So the labels
MediaPipe's Tasks API gives, and which side of the image each hand is on, have to be
measured again before vision/handedness.py's rules are trusted here.

Camera only: no ROS, no robot. A preview window (unmirrored: the operator's own view)
tells the operator what to do:
   5 s  get ready
  10 s  RIGHT hand only, moving about
  10 s  LEFT hand only, moving about
  12 s  BOTH hands, not crossed
and the raw detections (label, score, wrist x) are counted against that ground truth.
The clip and a per-frame log go to data/clips/ego_labels.avi / .jsonl for re-use.

Usage (with .venv-vision):
    tools/check_labels_ego.py [--cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision.hand_tracker import HandTracker  # noqa: E402

PHASES = (("ready", 5.0, "get ready: RIGHT hand first"), ("right", 10.0, "RIGHT hand only - move it around"),
          ("left", 10.0, "LEFT hand only - move it around"), ("both", 12.0, "BOTH hands, not crossed"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu", action="store_true", help="legacy MediaPipe API on the CPU instead of Tasks on the GPU")
    args = p.parse_args()

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    tracker = HandTracker(gpu=not args.cpu)
    clips = REPO / "data" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(clips / "ego_labels.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    log = (clips / "ego_labels.jsonl").open("w")
    stats: dict[str, list] = {name: [] for name, _, _ in PHASES[1:]}
    frames = Counter()
    t0 = time.monotonic()
    try:
        for name, length, text in PHASES:
            end = time.monotonic() + length
            while time.monotonic() < end:
                frame = np.asanyarray(pipe.wait_for_frames(3000).get_color_frame().get_data()).copy()
                t = time.monotonic() - t0
                hands = tracker.detect(frame, t)
                video.write(frame)
                rows = [{"label": h.label, "score": round(float(h.score), 3),
                         "x": round(float(h.image_landmarks[0, 0]), 3)} for h in hands]
                log.write(json.dumps({"t": round(t, 3), "phase": name, "hands": rows}) + "\n")
                if name != "ready":
                    frames[name] += 1
                    stats[name].append(rows)
                preview = frame.copy()
                for h in hands:
                    x, y = (h.image_landmarks[0, :2] * [640, 480]).astype(int)
                    cv2.putText(preview, f"{h.label} {h.score:.2f}", (x - 40, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(preview, f"{text}  ({end - time.monotonic():.0f}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("ego labels", preview)
                cv2.waitKey(1)
    finally:
        pipe.stop()
        video.release()
        log.close()
        cv2.destroyAllWindows()

    print(f"labels from {'legacy API (CPU)' if args.cpu else 'Tasks API (GPU)'}; tracker maps {tracker.label_to_side}")
    for side in ("right", "left"):
        dets = [d for rows in stats[side] for d in rows]
        found = sum(1 for rows in stats[side] if rows)
        c = Counter(d["label"] for d in dets)
        print(f"{side.upper()} hand only: found in {found}/{frames[side]} frames; labels {dict(c)}; "
              f"score median {np.median([d['score'] for d in dets]) if dets else float('nan'):.3f}; "
              f"wrist x median {np.median([d['x'] for d in dets]) if dets else float('nan'):.2f}")
    two = [rows for rows in stats["both"] if len(rows) == 2]
    if two:
        # ground truth from position: in the operator's own view the right hand is on the right
        agree = sum(1 for rows in two if max(rows, key=lambda d: d["x"])["label"] != min(rows, key=lambda d: d["x"])["label"])
        right_label = Counter(max(rows, key=lambda d: d["x"])["label"] for rows in two)
        print(f"BOTH: two hands in {len(two)}/{frames['both']} frames; different labels in {agree}; "
              f"label of the hand on the image right {dict(right_label)}")
    print(f"clip and log -> {clips / 'ego_labels.avi'}, .jsonl")


if __name__ == "__main__":
    main()
