#!/usr/bin/env python3
"""
hand_tracker_node.py — both hands from one camera, as ROS topics (docs/01_vision_design.md, 3.5).

Publishes, per side ("left", "right"), only on frames where that side was found:
  /<side>/allegro/vision/world_landmarks  Float32MultiArray (63), the hand repo's format.
                                          Its retargeter must run with --no-undo-mirror:
                                          these come from the unmirrored image.
  /<side>/wrist_pose                      PoseStamped in `camera_optical`; needs --camera-info.
and every frame:
  /camera/image_raw                       Image, bgr8, unmirrored.

All stamps are the time the frame was read.

A clip can stand in for the camera; it is replayed at its own frame rate.

Usage:
    vision/hand_tracker_node.py --source 0 --camera-info config/camera_hp_fhd.yaml
    vision/hand_tracker_node.py --source data/clips/v2_palm2.avi --camera-info <yaml>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import camera  # noqa: E402
from vision.hand_tracker import SIDES, HandResult, HandTracker  # noqa: E402

CAMERA_FRAME = "camera_optical"

# Same as the hand repo's publishers, so its subscribers match.
QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)


class HandTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("hand_tracker")
        self._landmarks = {s: self.create_publisher(Float32MultiArray, f"/{s}/allegro/vision/world_landmarks", QOS)
                           for s in SIDES}
        self._wrist = {s: self.create_publisher(PoseStamped, f"/{s}/wrist_pose", QOS) for s in SIDES}
        self._image = self.create_publisher(Image, "/camera/image_raw", QOS)
        self._bridge = CvBridge()

    def publish(self, frame: np.ndarray, hands: dict[str, HandResult], stamp) -> None:
        img = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img.header.stamp = stamp
        img.header.frame_id = CAMERA_FRAME
        self._image.publish(img)

        for side, hand in hands.items():
            self._landmarks[side].publish(Float32MultiArray(data=hand.world_landmarks.astype(np.float32).ravel().tolist()))
            if hand.wrist is None:
                continue
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = CAMERA_FRAME
            p = msg.pose.position
            p.x, p.y, p.z = (float(v) for v in hand.wrist.position)
            q = msg.pose.orientation
            q.x, q.y, q.z, q.w = (float(v) for v in Rotation.from_matrix(hand.wrist.rotation).as_quat())
            self._wrist[side].publish(msg)


def open_source(source: str, size: tuple[int, int]) -> tuple[cv2.VideoCapture, bool]:
    """(capture, is_clip). Cameras are asked for `size` in MJPG, as tools/calibrate_camera.py does."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        return cap, False
    if not Path(source).exists():
        sys.exit(f"[!] {source}: no such file")
    return cv2.VideoCapture(source), True


def pair(text: str) -> tuple[int, int]:
    a, b = text.lower().split("x")
    return int(a), int(b)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index (/dev/videoN) or a video file")
    p.add_argument("--camera-info", type=Path, help="intrinsics YAML; without it no wrist pose is published")
    p.add_argument("--size", type=pair, default=(640, 480),
                   help="camera capture size; ignored when --camera-info gives one")
    p.add_argument("--lite", action="store_true",
                   help="MediaPipe's lite landmark model: twice as fast, more jitter (design doc, V5)")
    args, ros_args = p.parse_known_args()

    calibration = camera.load(args.camera_info) if args.camera_info else None
    size = calibration.image_size if calibration else args.size
    cap, is_clip = open_source(args.source, size)

    rclpy.init(args=ros_args)
    node = HandTrackerNode()
    log = node.get_logger()
    if calibration is None:
        log.warn("no --camera-info: publishing landmarks only, no wrist pose")
    tracker = HandTracker(calibration, model_complexity=0 if args.lite else 1)

    period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_clip else 0.0
    t0 = time.monotonic()
    frames, seen, last_report = 0, {s: 0 for s in SIDES}, t0
    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                if is_clip:
                    log.info("end of clip")
                    break
                continue
            stamp = node.get_clock().now().to_msg()
            t = frames * period if is_clip else time.monotonic() - t0
            try:
                hands = tracker.process(frame, t)
            except ValueError as exc:  # frame size does not match the intrinsics
                log.error(str(exc))
                break
            node.publish(frame, hands, stamp)
            rclpy.spin_once(node, timeout_sec=0.0)

            frames += 1
            for side in hands:
                seen[side] += 1
            now = time.monotonic()
            if now - last_report >= 5.0:
                log.info(f"{frames / (now - t0):.1f} fps, frames with a hand: "
                         + ", ".join(f"{s} {seen[s]}/{frames}" for s in SIDES))
                last_report = now
            if is_clip:  # replay at the clip's own rate
                time.sleep(max(0.0, t0 + frames * period - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        cap.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
