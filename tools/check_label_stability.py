#!/usr/bin/env python3
"""
check_label_stability.py — design doc check V2: do the sides hold while both hands move?

Camera only: no ROS, no robot. Waits five seconds, then records fifteen, with a
mirrored preview whose yellow guides mark x = 0.2 and 0.8: V2 found MediaPipe's
labels wrong a third of the time outside them and never inside.

Without --crossed, the hands must not cross, so the operator's right hand is
always left of the left hand in the unmirrored image; that order is the ground
truth for the wrong-label counts. With --crossed there is none, and only the
frame-to-frame side swaps are reported.

The raw clip and a per-frame log go to data/clips/v2_<name>.avi / .jsonl, so
another hand model can be run on the same frames later.

Usage:
    tools/check_label_stability.py palm
    tools/check_label_stability.py back
    tools/check_label_stability.py cross --crossed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision.handedness import MEDIAPIPE_LABEL_TO_SIDE, MIN_SCORE, Detection, assign_sides  # noqa: E402

WAIT_S, RECORD_S = 5.0, 15.0
GUIDES = (0.2, 0.8)


def preview(frame: np.ndarray, text: str) -> None:
    """Mirrored for the operator; the guides are symmetric, so mirroring keeps them."""
    shown = cv2.flip(frame, 1)
    h, w = shown.shape[:2]
    for gx in GUIDES:
        cv2.line(shown, (int(gx * w), 0), (int(gx * w), h), (0, 255, 255), 2)
    cv2.putText(shown, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("V2", shown)
    cv2.waitKey(1)


def summarize(rows: list[dict], crossed: bool, seconds: float) -> None:
    n = len(rows)
    count = lambda f: sum(1 for r in rows if f(r))  # noqa: E731
    both = [r for r in rows if len(r["sides"]) == 2]
    x = lambda r, s: r["dets"][r["sides"][s]]["wrist_x"]  # noqa: E731

    print(f"frames={n}  ({n / seconds:.1f} fps)")
    print(f"  detected 2 / 1 / 0 hands : {count(lambda r: len(r['dets']) == 2)} / "
          f"{count(lambda r: len(r['dets']) == 1)} / {count(lambda r: not r['dets'])}")
    print(f"  assigned both / right only / left only / none : {len(both)} / "
          f"{count(lambda r: list(r['sides']) == ['right'])} / "
          f"{count(lambda r: list(r['sides']) == ['left'])} / {count(lambda r: not r['sides'])}")
    print(f"  detections dropped under {MIN_SCORE}: "
          f"{sum(1 for r in rows for d in r['dets'] if d['score'] < MIN_SCORE)}")
    print(f"  frames with the same label twice: "
          f"{count(lambda r: len(r['dets']) == 2 and r['dets'][0]['label'] == r['dets'][1]['label'])}")
    scores = [d["score"] for r in rows for d in r["dets"]]
    if scores:
        print(f"  score p5 / min: {np.percentile(scores, 5):.3f} / {min(scores):.3f}")

    swaps = 0
    for a, b in zip(rows, rows[1:]):
        if len(a["sides"]) == 2 and len(b["sides"]) == 2:
            keep = abs(x(b, "right") - x(a, "right")) + abs(x(b, "left") - x(a, "left"))
            swap = abs(x(b, "right") - x(a, "left")) + abs(x(b, "left") - x(a, "right"))
            swaps += swap < keep
    print(f"  frame-to-frame side swaps (nearest wrist says the sides traded): {swaps}")

    if crossed:
        return
    # No crossing: the leftmost hand in the unmirrored image is the right hand.
    wrong, by_x = [], {}
    for r in rows:
        if len(r["dets"]) != 2:
            continue
        for truth, d in zip(["right", "left"], sorted(r["dets"], key=lambda d: d["wrist_x"])):
            bad = MEDIAPIPE_LABEL_TO_SIDE[d["label"]] != truth
            if bad:
                wrong.append(d["score"])
            by_x.setdefault((truth, min(int(d["wrist_x"] * 5), 4)), []).append(bad)
    total = sum(len(v) for v in by_x.values())
    print(f"  wrong labels in two-hand frames: {len(wrong)} of {total}")
    if wrong:
        print(f"    their scores: median {np.median(wrong):.3f}, max {max(wrong):.3f}")
    for truth in ("right", "left"):
        cells = [f"x {b / 5:.1f}-{(b + 1) / 5:.1f}: {np.mean(by_x[(truth, b)]):.0%} of {len(by_x[(truth, b)])}"
                 for b in range(5) if (truth, b) in by_x]
        print(f"    {truth:5s} hand wrong by wrist x | " + " | ".join(cells))
    print(f"  assigned in the wrong order: "
          f"{sum(1 for r in both if x(r, 'right') > x(r, 'left'))} of {len(both)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="label for this run, e.g. palm, back, cross")
    p.add_argument("--crossed", action="store_true", help="the hands cross: no position ground truth")
    p.add_argument("--device", type=int, default=0, help="/dev/videoN")
    args = p.parse_args()

    clips = REPO / "data" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video = cv2.VideoWriter(str(clips / f"v2_{args.name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))

    t0 = time.monotonic()
    while time.monotonic() - t0 < WAIT_S:
        ok, frame = cap.read()
        if ok:
            preview(frame, f"get ready {WAIT_S - (time.monotonic() - t0):.0f}")

    rows = []
    with open(clips / f"v2_{args.name}.jsonl", "w") as log:
        t0 = time.monotonic()
        while time.monotonic() - t0 < RECORD_S:
            ok, frame = cap.read()
            if not ok:
                continue
            video.write(frame)
            preview(frame, f"REC {RECORD_S - (time.monotonic() - t0):.0f}")
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # unmirrored
            dets = [Detection(h.classification[0].label, h.classification[0].score, lm.landmark[0].x)
                    for lm, h in zip(res.multi_hand_landmarks or [], res.multi_handedness or [])]
            row = {"t": round(time.monotonic() - t0, 3), "dets": [d._asdict() for d in dets],
                   "sides": assign_sides(dets)}
            rows.append(row)
            log.write(json.dumps(row) + "\n")
    cap.release()
    video.release()
    cv2.destroyAllWindows()

    print(args.name, end=": ")
    summarize(rows, args.crossed, RECORD_S)
    print(f"  clip: {clips / f'v2_{args.name}.avi'}")


if __name__ == "__main__":
    main()
