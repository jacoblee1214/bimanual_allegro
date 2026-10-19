#!/usr/bin/env python3
"""
hand_tracker_node_v3.py — both hands from one camera, as ROS topics (docs/01_vision_design.md, 3.5).

v2 adds the egocentric RealSense setup (docs/05_ego_camera.md):
  --source realsense   the RealSense colour stream, with the device's factory intrinsics
  --view ego           sides by image position (vision/handedness_ego.py): from above and
                       behind, MediaPipe's labels were right only 70-82% of the time.
For a recorded ego clip, pass --view ego and --camera-info config/ego_camera_info.yaml.

v3 adds --ray-filter (with --view ego): the wrist position is filtered along the camera
ray by depth_filter.RayKalman instead of One Euro in every direction
(vision/hand_tracker_ego_v2.py, docs/results/m4_depth_filter.md).

Publishes, per side ("left", "right"), only on frames where that side was found:
  /<side>/allegro/vision/world_landmarks  Float32MultiArray (63), the hand repo's format.
                                          Its retargeter must run with --no-undo-mirror:
                                          these come from the unmirrored image.
  /<side>/wrist_pose                      PoseStamped in `camera_optical`; needs --camera-info.
and every frame:
  /camera/image_raw                       Image, bgr8, unmirrored.

All stamps are the time the frame was read.

A clip can stand in for the camera; it is replayed at its own frame rate, once or with
--loop over and over (the simulation launch uses that).

Usage:
    vision/hand_tracker_node.py --source 0 --camera-info config/camera_hp_fhd.yaml --gpu
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
from vision.hand_tracker_ego import EgoHandTracker  # noqa: E402
from vision.hand_tracker_ego_v2 import RayEgoHandTracker  # noqa: E402

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


class RealSenseColour:
    """The RealSense colour stream behind cv2.VideoCapture's read()/get()/release()."""

    def __init__(self, size: tuple[int, int], fps: int = 30) -> None:
        import pyrealsense2 as rs
        self._pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, fps)
        intr = self._pipe.start(cfg).get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
        # The colour stream's distortion is inverse Brown-Conrady, the other direction from
        # OpenCV's model, so its coefficients cannot be passed on as they are. They are small
        # (about 0.6% at the image corner on the lab's D455), so the distortion is left out.
        self.calibration = camera.CameraCalibration((intr.width, intr.height), K, np.zeros(5))
        self.fps = fps

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self._pipe.wait_for_frames(3000).get_color_frame()
        return (True, np.asanyarray(frame.get_data()).copy()) if frame else (False, None)

    def get(self, prop: int) -> float:
        return float(self.fps) if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self._pipe.stop()


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
    p.add_argument("--source", default="0", help="camera index (/dev/videoN), 'realsense', or a video file")
    p.add_argument("--view", choices=["front", "ego"], default="front",
                   help="front: operator faces the camera (labels decide sides); ego: camera above and behind")
    p.add_argument("--ray-filter", action="store_true",
                   help="with --view ego: filter the wrist position along the camera ray (RayKalman)")
    p.add_argument("--camera-info", type=Path, help="intrinsics YAML; without it no wrist pose is published")
    p.add_argument("--size", type=pair, default=(640, 480),
                   help="camera capture size; ignored when --camera-info gives one")
    model = p.add_mutually_exclusive_group()
    model.add_argument("--lite", action="store_true",
                       help="MediaPipe's lite landmark model: twice as fast, more jitter (design doc, V5)")
    model.add_argument("--gpu", action="store_true",
                       help="MediaPipe's Tasks API on the GPU: 4 ms a frame instead of 30 (design doc, V5)")
    p.add_argument("--loop", action="store_true", help="replay a clip from the start when it ends")
    args, ros_args = p.parse_known_args()

    calibration = camera.load(args.camera_info) if args.camera_info else None
    size = calibration.image_size if calibration else args.size
    if args.source == "realsense":
        cap, is_clip = RealSenseColour(size), False
        calibration = calibration or cap.calibration
    else:
        cap, is_clip = open_source(args.source, size)

    rclpy.init(args=ros_args)
    node = HandTrackerNode()
    log = node.get_logger()
    if calibration is None:
        log.warn("no --camera-info: publishing landmarks only, no wrist pose")
    if args.ray_filter and args.view != "ego":
        sys.exit("[!] --ray-filter needs --view ego")
    tracker_class = (RayEgoHandTracker if args.ray_filter else EgoHandTracker) if args.view == "ego" else HandTracker
    tracker = tracker_class(calibration, model_complexity=0 if args.lite else 1, gpu=args.gpu)
    log.info(f"source {args.source}, {args.view} view, {calibration.camera_matrix[0, 0]:.0f} px focal length"
             if calibration is not None else f"source {args.source}, {args.view} view")

    period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_clip else 0.0
    t0 = time.monotonic()
    frames, seen, last_report = 0, {s: 0 for s in SIDES}, t0
    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                if is_clip and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
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
