#!/usr/bin/env python3
"""
demo_tracker_node.py — vision/hand_tracker_node_v4.py with a picture for the audience.

Same arguments and topics as node v4, plus
  /demo/image   Image, bgr8: the camera frame with every detected hand's landmarks drawn,
                and a status line per side (tracking or not, and with --depth where the
                wrist distance came from)
The banner says whether real arms are moving (--banner "REAL ARMS"), so nobody in front
of the camera has to guess.

The landmarks are MediaPipe's image landmarks, which the trackers do not hand out; this
keeps the last detection by wrapping HandTracker.detect, which every tracker calls once a
frame.

Usage (with .venv-vision; launch/demo.launch.py starts it):
    demo/demo_tracker_node.py --source realsense --view ego --depth --fusion v2 --gpu
    demo/demo_tracker_node.py --source 0 --camera-info config/camera_nominal_640x480.yaml --gpu
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import hand_tracker_node_v4 as v4  # noqa: E402
from vision.hand_tracker import SIDES, HandTracker  # noqa: E402

# MediaPipe's 21-point hand: wrist, then four points per finger, thumb first.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
GREEN, RED, WHITE, YELLOW = (80, 220, 80), (60, 60, 230), (255, 255, 255), (60, 220, 230)

_last_raw: list = []
_detect = HandTracker.detect


def _keep_detect(self, frame_bgr, t):
    raw = _detect(self, frame_bgr, t)
    _last_raw[:] = raw
    return raw


HandTracker.detect = _keep_detect


def _banner_arg() -> str:
    """--banner TEXT, taken out of sys.argv before node v4 parses it."""
    if "--banner" not in sys.argv:
        return "SIMULATION"
    i = sys.argv.index("--banner")
    text = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return text


BANNER = _banner_arg()


def annotate(frame, hands) -> None:
    h, w = frame.shape[:2]
    for raw in _last_raw:
        pts = [(int(x * w), int(y * h)) for x, y, _ in raw.image_landmarks]
        for a, b in BONES:
            cv2.line(frame, pts[a], pts[b], WHITE, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, YELLOW, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (w, 34), (30, 30, 30), -1)
    cv2.putText(frame, BANNER, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED if "REAL" in BANNER else WHITE, 2,
                cv2.LINE_AA)
    for k, side in enumerate(SIDES):
        hand = hands.get(side)
        if hand is None:
            text, colour = f"{side}: not seen", RED
        else:
            fused = getattr(hand, "fused", None)
            text = f"{side}: tracking" + (f" ({fused.source})" if fused is not None else "")
            colour = GREEN if hand.wrist is not None else YELLOW
        cv2.putText(frame, text, (8 + k * w // 2, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


class DemoTrackerNode(v4.DepthHandTrackerNode):
    def __init__(self) -> None:
        super().__init__()
        self._demo = self.create_publisher(Image, "/demo/image", v4.QOS)

    def publish(self, frame, hands, stamp) -> None:
        try:
            super().publish(frame, hands, stamp)
            shown = frame.copy()
            annotate(shown, hands)
            img = self._bridge.cv2_to_imgmsg(shown, encoding="bgr8")
            img.header.stamp = stamp
            self._demo.publish(img)
        except RCLError:
            # Ctrl-C shuts the context down between frames; node v4's loop then ends by itself.
            if rclpy.ok():
                raise


if __name__ == "__main__":
    v4.DepthHandTrackerNode = DemoTrackerNode
    v4.main()
