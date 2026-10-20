#!/usr/bin/env python3
"""
eval_node_v4_ros.py — milestone M6: hand_tracker_node_v4.py end to end over ROS, on a recorded clip.

Starts the node as a child process,
    vision/hand_tracker_node_v4.py --source clip:data/clips/<clip> --view ego --depth --fusion v1 --gpu
(no --loop: it replays the clip once at its recorded rate and exits), listens to every
topic it publishes, and then:

  rates     messages per topic, rate and gaps between arrivals (received without
            deserialising; `ros2 topic hz` cannot keep up with 900 kB images)
  same      the /<side>/wrist_pose positions and /<side>/wrist_depth_diag sources, in
            order, against vision/hand_tracker_ego_v3.py run here on the same clip without
            ROS (MediaPipe on the GPU, meta.json intrinsics, as the node)
  timing    the node's own per-frame log lines (read / track / publish, p50 / p95 ms over
            each 5 s)

Needs the ROS environment. Use a ROS domain of its own so nothing else sees it, e.g.
    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    .venv-vision/bin/python -B tools/eval_node_v4_ros.py [ego_depth_424]
Writes data/results/m6_node_v4_ros.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision.hand_tracker import SIDES  # noqa: E402
from vision.hand_tracker_ego_v3 import DepthEgoHandTracker  # noqa: E402
from vision.rgbd_source import ClipRGBD  # noqa: E402

OUT = REPO / "data" / "results" / "m6_node_v4_ros.json"
QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
TOPICS = {"/camera/image_raw": Image}
for _s in SIDES:
    TOPICS |= {f"/{_s}/wrist_pose": PoseStamped, f"/{_s}/wrist_depth_diag": String,
               f"/{_s}/allegro/vision/world_landmarks": Float32MultiArray}
LOG_LINE = re.compile(r"([\d.]+) fps.*read ([\d.]+)/([\d.]+), track ([\d.]+)/([\d.]+), publish ([\d.]+)/([\d.]+)")


def offline(clip: Path) -> dict:
    src = ClipRGBD(clip)
    tracker = DepthEgoHandTracker(src.calibration, gpu=True)
    out = {s: {"p": [], "source": []} for s in SIDES}
    while True:
        ok, frame, depth = src.read()
        if not ok:
            break
        for side, h in tracker.process(frame, src.t - src.times[0], depth).items():
            if h.wrist is not None:
                out[side]["p"].append(h.wrist.position)
                out[side]["source"].append(h.fused.source)
    tracker.close()
    src.release()
    return out


def rate(stamps: list[float]) -> dict:
    if len(stamps) < 2:
        return {"n": len(stamps)}
    d = np.diff(stamps)
    return {"n": len(stamps), "rate_hz": round(len(d) / (stamps[-1] - stamps[0]), 2),
            "gap_p50_ms": round(float(np.median(d)) * 1000, 1), "gap_p95_ms": round(float(np.percentile(d, 95)) * 1000, 1),
            "gap_max_ms": round(float(d.max()) * 1000, 1)}


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
    node = rclpy.create_node("eval_node_v4_ros")
    got: dict[str, list] = {t: [] for t in TOPICS}
    for topic, kind in TOPICS.items():
        node.create_subscription(kind, topic, lambda m, t=topic: got[t].append((time.monotonic(), m)), QOS, raw=True)

    log_path = args.out.with_suffix(".node.log")
    cmd = [sys.executable, "-B", str(REPO / "vision" / "hand_tracker_node_v4.py"), "--source", f"clip:{clip}",
           "--view", "ego", "--depth", "--fusion", "v1", "--gpu"]
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
    same = {}
    for s in SIDES:
        poses = [deserialize_message(raw, PoseStamped) for _, raw in got[f"/{s}/wrist_pose"]]
        p = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z] for m in poses])
        sources = [json.loads(deserialize_message(raw, String).data)["source"] for _, raw in got[f"/{s}/wrist_depth_diag"]]
        q = np.array(ref[s]["p"])
        entry = {"published": len(p), "offline": len(q), "diag_published": len(sources)}
        if len(p) == len(q):
            entry["position_max_diff_mm"] = round(float(np.abs(p - q).max()) * 1000, 6)
        entry["diag_sources_equal"] = sources == ref[s]["source"]
        entry["sensor_share_published"] = round(float(np.mean(np.array(sources) == "depth")), 4) if sources else None
        same[s] = entry
    payload = {
        "meta": {"date": datetime.date.today().isoformat(), "tool": "tools/eval_node_v4_ros.py", "clip": str(clip.relative_to(REPO)),
                 "command": "vision/hand_tracker_node_v4.py " + " ".join(cmd[3:]),
                 "ros_domain_id": domain, "rmw": os.environ.get("RMW_IMPLEMENTATION", "(default)"),
                 "node_exit": child.returncode, "node_log": str(log_path.relative_to(REPO))},
        "rates": {t: rate([s for s, _ in got[t]]) for t in TOPICS},
        "same_as_offline": same,
        "node_timing_lines": timing,
        "node_timing_summary": {k: {"median": round(float(np.median([x[k] for x in timing])), 2),
                                    "max": round(float(np.max([x[k] for x in timing])), 2)}
                                for k in (timing[0] if timing else {})},
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("rates", "same_as_offline", "node_timing_summary")}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
