#!/usr/bin/env python3
"""
calibrate_camera.py — checkerboard calibration for the hand-tracking camera.

Hold a printed checkerboard in view and press SPACE to keep a view whenever its
corners are drawn. Vary it: near and far, every corner of the image, tilted
30-45 degrees each way. C calibrates and saves once there are enough views,
Q quits.

The intrinsics only hold for the resolution they were taken at, so calibrate at
the one the tracker will run at. The image is shown unmirrored, as the tracker
sees it (docs/01_vision_design.md, 3.1).

Usage:
    tools/calibrate_camera.py --device 0 --size 1280x720 --board 9x6 --name hp_fhd
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import camera  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def pair(text: str) -> tuple[int, int]:
    a, b = text.lower().split("x")
    return int(a), int(b)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0, help="/dev/videoN")
    p.add_argument("--size", type=pair, default=(1280, 720), help="WxH to capture and calibrate at")
    p.add_argument("--board", type=pair, default=(9, 6), help="inner corners, COLSxROWS")
    p.add_argument("--name", default="hp_fhd", help="camera name; saved as config/camera_<name>.yaml")
    p.add_argument("--min-views", type=int, default=15)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    # MJPG: this webcam offers 1280x720 and up only compressed.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.size[1])
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"[!] /dev/video{args.device}: no frame")
    got = (frame.shape[1], frame.shape[0])
    if got != args.size:
        sys.exit(f"[!] asked for {args.size[0]}x{args.size[1]}, the camera gives {got[0]}x{got[1]}")

    out = REPO / "config" / f"camera_{args.name}.yaml"
    views: list[np.ndarray] = []
    print(f"[i] {got[0]}x{got[1]}, board {args.board[0]}x{args.board[1]} inner corners -> {out}")
    print("[i] SPACE keep view   C calibrate and save   Q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, args.board, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        )
        shown = frame.copy()
        if found:
            cv2.drawChessboardCorners(shown, args.board, corners, found)
        for c in views:  # where earlier views sat, to see what is still uncovered
            cv2.circle(shown, tuple(int(v) for v in c.reshape(-1, 2).mean(axis=0)), 6, (0, 200, 255), -1)
        cv2.putText(shown, f"views {len(views)}/{args.min_views}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibrate_camera", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            views.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX))
            print(f"[+] view {len(views)}")
        elif key == ord("c"):
            if len(views) < args.min_views:
                print(f"[!] {len(views)} views, need {args.min_views}")
                continue
            cal, rms = camera.calibrate(views, args.board, got)
            note = (f"{args.name}, {got[0]}x{got[1]}, {date.today()}, tools/calibrate_camera.py\n"
                    f"{len(views)} views of a {args.board[0]}x{args.board[1]} board, RMS {rms:.3f} px")
            camera.save(cal, out, args.name, note)
            print(f"[✓] RMS {rms:.3f} px -> {out}")
            print(np.array2string(cal.camera_matrix, precision=1, suppress_small=True))
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
