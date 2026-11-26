#!/usr/bin/env python3
"""
demo_tracker_node_v3.py — demo_tracker_node_v2.py with the sensor palm rotation held over
frames without one (vision/rotation_hold.py), for following the wrist's rotation.

Same picture, topics and arguments as v2. Only the orientation in /<side>/wrist_pose
changes, and only on frames whose depth fusion source is "hold" or "pnp".

Usage (with .venv-vision; launch/teleop_real_v2.launch.py starts it):
    demo/demo_tracker_node_v3.py --source realsense --view ego --depth --fusion v3 --gpu
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo.demo_tracker_node as v1  # noqa: E402  (wraps HandTracker.detect, takes --banner)
from vision import hand_tracker_node_v5 as v5  # noqa: E402
from vision.rotation_hold import hold_rotation  # noqa: E402

if __name__ == "__main__":
    v5.DepthHandTrackerNode = v1.DemoTrackerNode
    plain = v5.fusion_factory
    v5.fusion_factory = lambda name: hold_rotation(plain(name))
    v5.main()
