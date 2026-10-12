#!/usr/bin/env python3
"""
check_labels.py — design doc check V1: does MediaPipe label the hands as expected?

Camera only: no ROS, no robot. Runs MediaPipe on the unmirrored image for ten
seconds after a five-second wait, and reports its raw labels and the sides
vision/handedness.py assigns from them. Raise one hand (or both) before it
starts. Saves one annotated frame to data/clips/v1_<name>.jpg.

Usage:
    tools/check_labels.py right_only        # then left_only, both
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision.handedness import Detection, assign_sides  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="label for this run, e.g. right_only")
    p.add_argument("--device", type=int, default=0, help="/dev/videoN")
    args = p.parse_args()

    clips = REPO / "data" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:  # time to get the hand up
        cap.read()

    frames, raw, sides, scores, xs, saved = 0, Counter(), Counter(), {}, {}, False
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10.0:
        ok, frame = cap.read()
        if not ok:
            continue
        frames += 1
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # unmirrored
        dets = []
        for lm, hd in zip(res.multi_hand_landmarks or [], res.multi_handedness or []):
            c = hd.classification[0]
            dets.append(Detection(c.label, c.score, lm.landmark[0].x))
            raw[c.label] += 1
            scores.setdefault(c.label, []).append(c.score)
            xs.setdefault(c.label, []).append(lm.landmark[0].x)
        assigned = assign_sides(dets)
        sides[tuple(sorted(assigned))] += 1
        if not saved and dets and time.monotonic() - t0 > 3.0:
            side_of = {i: s for s, i in assigned.items()}
            shot = frame.copy()
            for i, (lm, d) in enumerate(zip(res.multi_hand_landmarks, dets)):
                mp.solutions.drawing_utils.draw_landmarks(shot, lm, mp.solutions.hands.HAND_CONNECTIONS)
                cv2.putText(shot, f"MP={d.label} {d.score:.2f} -> {side_of.get(i, 'dropped')}",
                            (int(d.wrist_x * 640) - 80, int(lm.landmark[0].y * 480) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(clips / f"v1_{args.name}.jpg"), shot)
            saved = True
    cap.release()

    print(f"{args.name}: frames={frames}")
    print("MediaPipe raw labels (per detection):", dict(raw))
    for k in scores:
        print(f"  {k}: score mean {np.mean(scores[k]):.3f} min {np.min(scores[k]):.3f}, "
              f"wrist_x mean {np.mean(xs[k]):.2f}")
    print("assigned sides (per frame):", {(",".join(k) or "none"): v for k, v in sides.items()})


if __name__ == "__main__":
    main()
