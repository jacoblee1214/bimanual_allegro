#!/usr/bin/env python3
"""
eval_node_v5_ros.py — milestone M7: hand_tracker_node_v5.py --fusion v3 end to end over ROS, on a recorded clip.

As tools/eval_node_v4_ros.py (whose topic list, QoS, log parser and rate summary are
imported), for the v5 node with the v3 fusion:
    vision/hand_tracker_node_v5.py --source clip:data/clips/<clip> --view ego --depth --fusion v3
        --camera-info config/ego_camera_info.yaml --gpu
It replays the clip once at its recorded rate and exits. Then:

  rates     messages per topic
  same      /<side>/wrist_pose positions and /<side>/wrist_depth_diag sources and pixels, in
            order, against vision/hand_tracker_ego_v3.py with WristDepthV3 run here on the
            same clip without ROS (MediaPipe on the GPU, the same intrinsics)
  diag      sources published, and how often `pixels` is filled (it was always null in v4
            with v2)
  timing    the node's own per-frame log lines

Needs the ROS environment, in a domain of its own:
    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    .venv-vision/bin/python -B tools/eval_node_v5_ros.py [ego_depth_424]
Writes data/results/m7_node_v5_ros.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.eval_node_v4_ros import LOG_LINE, QOS, TOPICS, rate  # noqa: E402
from vision import camera  # noqa: E402
from vision.depth_wrist_v3 import WristDepthV3  # noqa: E402
from vision.hand_tracker import SIDES  # noqa: E402
from vision.hand_tracker_ego_v3 import DepthEgoHandTracker  # noqa: E402
from vision.rgbd_source import ClipRGBD  # noqa: E402

OUT = REPO / "data" / "results" / "m7_node_v5_ros.json"
CAMERA_INFO = REPO / "config" / "ego_camera_info.yaml"


def offline(clip: Path) -> dict:
    src = ClipRGBD(clip)
    tracker = DepthEgoHandTracker(camera.load(CAMERA_INFO), fusion_factory=WristDepthV3, gpu=True)
    out = {s: {"p": [], "source": [], "pixels": []} for s in SIDES}
    while True:
        ok, frame, depth = src.read()
        if not ok:
            break
        for side, h in tracker.process(frame, src.t - src.times[0], depth).items():
            if h.wrist is not None:
                out[side]["p"].append(h.wrist.position)
                out[side]["source"].append(h.fused.source)
                out[side]["pixels"].append(h.fused.pixels)
    tracker.close()
    src.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?", default="ego_depth_424")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds before the node is stopped")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    domain = os.environ.get("ROS_DOMAIN_ID", "0")
    if domain in ("0", "42"):
        sys.exit(f"[!] ROS_DOMAIN_ID={domain}: use a domain of its own (see the docstring)")
    clip = REPO / "data" / "clips" / args.clip

    rclpy.init()
    node = rclpy.create_node("eval_node_v5_ros")
    got: dict[str, list] = {t: [] for t in TOPICS}
    for topic, kind in TOPICS.items():
        node.create_subscription(kind, topic, lambda m, t=topic: got[t].append((time.monotonic(), m)), QOS, raw=True)

    log_path = args.out.with_suffix(".node.log")
    cmd = [sys.executable, "-B", str(REPO / "vision" / "hand_tracker_node_v5.py"), "--source", f"clip:{clip}",
           "--view", "ego", "--depth", "--fusion", "v3", "--camera-info", str(CAMERA_INFO), "--gpu"]
    with log_path.open("w") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        end = time.monotonic() + args.timeout
        while child.poll() is None and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
        if child.poll() is None:  # only the child started here
            child.send_signal(signal.SIGINT)
            child.wait(10)
        for _ in range(50):  # whatever is still in flight
            rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_node()
    rclpy.shutdown()

    timing = [dict(zip(("fps", "read_p50", "read_p95", "track_p50", "track_p95", "publish_p50", "publish_p95"),
                       map(float, m.groups())))
              for m in map(LOG_LINE.search, log_path.read_text().splitlines()) if m]
    ref = offline(clip)
    same, diag = {}, {}
    for s in SIDES:
        poses = [deserialize_message(raw, PoseStamped) for _, raw in got[f"/{s}/wrist_pose"]]
        p = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z] for m in poses])
        msgs = [json.loads(deserialize_message(raw, String).data) for _, raw in got[f"/{s}/wrist_depth_diag"]]
        sources, pixels = [m["source"] for m in msgs], [m["pixels"] for m in msgs]
        q = np.array(ref[s]["p"])
        entry = {"published": len(p), "offline": len(q), "diag_published": len(msgs)}
        if len(p) == len(q) and len(p):
            entry["position_max_diff_mm"] = round(float(np.abs(p - q).max()) * 1000, 6)
        entry["diag_sources_equal"] = sources == ref[s]["source"]
        entry["diag_pixels_equal"] = pixels == ref[s]["pixels"]
        same[s] = entry
        src_arr = np.array(sources)
        has_px = np.array([x is not None for x in pixels], dtype=bool)
        diag[s] = {
            "by_source": {k: int(c) for k, c in zip(*np.unique(src_arr, return_counts=True))} if msgs else {},
            "pixels_filled": {k: round(float(has_px[src_arr == k].mean()), 4) for k in np.unique(src_arr)} if msgs else {},
            "pixels_on_depth_frames": {
                "p5": int(np.percentile([x for x, k in zip(pixels, sources) if k == "depth"], 5)),
                "p50": int(np.percentile([x for x, k in zip(pixels, sources) if k == "depth"], 50))}
            if "depth" in sources else None,
        }
    payload = {
        "meta": {"date": datetime.date.today().isoformat(), "tool": "tools/eval_node_v5_ros.py",
                 "clip": str(clip.relative_to(REPO)),
                 "command": "vision/hand_tracker_node_v5.py " + " ".join(cmd[3:]),
                 "ros_domain_id": domain, "rmw": os.environ.get("RMW_IMPLEMENTATION", "(default)"),
                 "node_exit": child.returncode, "node_log": str(log_path.relative_to(REPO))},
        "rates": {t: rate([s for s, _ in got[t]]) for t in TOPICS},
        "same_as_offline": same,
        "diag": diag,
        "node_timing_lines": timing,
        "node_timing_summary": {k: {"median": round(float(np.median([x[k] for x in timing])), 2),
                                    "max": round(float(np.max([x[k] for x in timing])), 2)}
                                for k in (timing[0] if timing else {})},
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("rates", "same_as_offline", "diag", "node_timing_summary")}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
