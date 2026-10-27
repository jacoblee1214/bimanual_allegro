#!/usr/bin/env python3
"""
detect_camera.py — which camera to run the demo with; prints one line for demo/run_demo.sh.

  realsense depth    a RealSense on USB 3: egocentric view with the depth sensor
  realsense          a RealSense on USB 2: colour only (colour + depth do not keep up)
  <N>                /dev/videoN, the first other camera that gives colour frames
  none

RealSense first, since the demo's camera geometry (config/ego_camera.yaml) is for it.
Other cameras are tried in /dev/video order; infrared cameras (grey frames) and the
RealSense's own nodes are skipped.

Usage (with .venv-vision):
    demo/detect_camera.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def realsense() -> str | None:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None
    devices = rs.context().query_devices()
    if len(devices) == 0:
        return None
    usb = devices[0].get_info(rs.camera_info.usb_type_descriptor)
    return "realsense depth" if usb.startswith("3") else "realsense"


def webcam() -> str | None:
    nodes = sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:]))
    for node in nodes:
        if "RealSense" in (node / "name").read_text():
            continue
        index = int(node.name[5:])
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if not ok or frame is None or frame.ndim != 3:
            continue
        f = frame.astype(np.int16)
        if np.abs(f[..., 0] - f[..., 1]).mean() + np.abs(f[..., 1] - f[..., 2]).mean() < 2.0:
            continue  # grey: an infrared camera
        return str(index)
    return None


if __name__ == "__main__":
    print(realsense() or webcam() or "none")
