#!/usr/bin/env python3
"""
hand_tracker_node_v5.py — hand_tracker_node_v4.py with a third depth fusion, `--fusion v3`
(docs/results/m7_hold_v3.md).

  --fusion v1|v2|v3  v1: M5's palm depth (vision/depth_wrist.py); v2: vision/depth_wrist_v2.py
                     WristDepthV2; v3: vision/depth_wrist_v3.py WristDepthV3, v2's sensor
                     palm plane with a hold that keeps the last sensor range on the current
                     pixel ray instead of PnP x the recent ratios. v2 and v3 are imported only
                     when asked for.

Everything else, topics included, is v4's: the node class and the timing log are imported
from it, and main() is v4's with the extra choice (v4's main() builds its parser and picks
the fusion itself, so it cannot be reused as it is). With v3, /<side>/wrist_depth_diag also
carries `pixels` (sensor pixels palm_depth kept, null on frames without a sensor value);
v4's node already published it when the fusion's output had the field.

Usage:
    vision/hand_tracker_node_v5.py --source realsense --view ego --depth --fusion v3 --gpu
    vision/hand_tracker_node_v5.py --source clip:data/clips/ego_depth_424 --view ego --depth --fusion v3 \
        --camera-info config/ego_camera_info.yaml --gpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import camera  # noqa: E402
from vision.hand_tracker import SIDES, HandTracker  # noqa: E402
from vision.hand_tracker_ego import EgoHandTracker  # noqa: E402
from vision.hand_tracker_ego_v2 import RayEgoHandTracker  # noqa: E402
from vision.hand_tracker_ego_v3 import DepthEgoHandTracker  # noqa: E402
from vision.hand_tracker_node_v3 import RealSenseColour, open_source, pair  # noqa: E402
from vision.hand_tracker_node_v4 import CLIP_PREFIX, DepthHandTrackerNode, Timings  # noqa: E402
from vision.hand_tracker_node_v4 import fusion_factory as fusion_factory_v4  # noqa: E402
from vision.rgbd_source import ClipRGBD, RealSenseRGBD  # noqa: E402


def fusion_factory(name: str):
    if name != "v3":
        return fusion_factory_v4(name)
    try:
        from vision.depth_wrist_v3 import WristDepthV3
    except ImportError as exc:
        sys.exit(f"[!] --fusion v3 needs vision/depth_wrist_v3.py (WristDepthV3): {exc}")
    return WristDepthV3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0",
                   help="camera index (/dev/videoN), 'realsense', a video file, or clip:<record_ego_depth dir>")
    p.add_argument("--view", choices=["front", "ego"], default="front",
                   help="front: operator faces the camera (labels decide sides); ego: camera above and behind")
    p.add_argument("--ray-filter", action="store_true",
                   help="with --view ego: filter the wrist position along the camera ray (RayKalman)")
    p.add_argument("--depth", action="store_true",
                   help="with --view ego and --source realsense or clip:<dir>: wrist distance from the depth sensor")
    p.add_argument("--fusion", choices=["v1", "v2", "v3"], default="v1", help="with --depth: which depth fusion")
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

    is_rgbd_clip = args.source.startswith(CLIP_PREFIX)
    if args.ray_filter and args.view != "ego":
        sys.exit("[!] --ray-filter needs --view ego")
    if args.depth and args.view != "ego":
        sys.exit("[!] --depth needs --view ego")
    if args.depth and not (args.source == "realsense" or is_rgbd_clip):
        sys.exit("[!] --depth needs --source realsense or clip:<dir>")

    calibration = camera.load(args.camera_info) if args.camera_info else None
    size = calibration.image_size if calibration else args.size
    if is_rgbd_clip:
        clip_dir = Path(args.source[len(CLIP_PREFIX):])
        if not (clip_dir / "meta.json").exists():
            sys.exit(f"[!] {clip_dir}: not a tools/record_ego_depth.py clip")
        cap, is_clip = ClipRGBD(clip_dir, depth=args.depth), True
        calibration = calibration or cap.calibration
    elif args.source == "realsense":
        cap, is_clip = (RealSenseRGBD() if args.depth else RealSenseColour(size)), False
        calibration = calibration or cap.calibration
    else:
        cap, is_clip = open_source(args.source, size)

    rclpy.init(args=ros_args)
    node = DepthHandTrackerNode()
    log = node.get_logger()
    if calibration is None:
        log.warn("no --camera-info: publishing landmarks only, no wrist pose")
    detector = dict(model_complexity=0 if args.lite else 1, gpu=args.gpu)
    if args.depth:
        tracker = DepthEgoHandTracker(calibration, fusion_factory=fusion_factory(args.fusion), **detector)
    else:
        tracker_class = (RayEgoHandTracker if args.ray_filter else EgoHandTracker) if args.view == "ego" else HandTracker
        tracker = tracker_class(calibration, **detector)
    log.info(f"source {args.source}, {args.view} view"
             + (f", {calibration.camera_matrix[0, 0]:.0f} px focal length" if calibration is not None else "")
             + (f", depth fusion {args.fusion}" if args.depth else ""))

    # Clip time: an RGBD clip's recorded times, a video's frame count over its rate; both
    # shifted by `offset` on every loop so that time keeps increasing for the filters.
    period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_clip and not is_rgbd_clip else 0.0
    t0 = time.monotonic()
    frames, seen, last_report, offset = 0, {s: 0 for s in SIDES}, t0, 0.0
    clip_frames = 0
    timings = Timings()
    try:
        while rclpy.ok():
            r0 = time.perf_counter()
            if is_rgbd_clip or isinstance(cap, RealSenseRGBD):
                ok, frame, depth = cap.read()
            else:
                (ok, frame), depth = cap.read(), None
            r1 = time.perf_counter()
            if not ok:
                if is_clip and args.loop:
                    if is_rgbd_clip:
                        offset += cap.times[-1] - cap.times[0] + float(np.median(np.diff(cap.times)))
                        cap.rewind()
                    else:
                        offset += clip_frames * period
                        clip_frames = 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if is_clip:
                    log.info("end of clip")
                    break
                continue
            if is_rgbd_clip:
                t = offset + cap.t - cap.times[0]
            elif is_clip:
                t = offset + clip_frames * period
            else:
                t = time.monotonic() - t0
            if is_clip:  # replay at the clip's own rate
                time.sleep(max(0.0, t0 + t - time.monotonic()))
            stamp = node.get_clock().now().to_msg()
            r2 = time.perf_counter()
            try:
                hands = tracker.process(frame, t, depth) if args.depth else tracker.process(frame, t)
            except ValueError as exc:  # frame or depth size does not match the intrinsics
                log.error(str(exc))
                break
            timings.read.append((r1 - r0) * 1000.0)
            r3 = time.perf_counter()
            timings.track.append((r3 - r2) * 1000.0)
            node.publish(frame, hands, stamp)
            timings.publish.append((time.perf_counter() - r3) * 1000.0)
            rclpy.spin_once(node, timeout_sec=0.0)

            frames += 1
            clip_frames += 1
            for side in hands:
                seen[side] += 1
            now = time.monotonic()
            if now - last_report >= 5.0:
                log.info(f"{frames / (now - t0):.1f} fps, frames with a hand: "
                         + ", ".join(f"{s} {seen[s]}/{frames}" for s in SIDES) + "; " + timings.report())
                last_report = now
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
