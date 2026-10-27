#!/usr/bin/env python3
"""
demo_tracker_node_v2.py — demo_tracker_node.py on vision/hand_tracker_node_v5.py, so the
demo can use depth fusion v3 (`--fusion v3`, docs/results/m7_hold_v3.md).

Same picture (/demo/image) and arguments as v1, plus node v5's `--fusion v3`.

Usage (with .venv-vision; launch/demo_v2.launch.py starts it):
    demo/demo_tracker_node_v2.py --source realsense --view ego --depth --fusion v3 --gpu
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo.demo_tracker_node as v1  # noqa: E402  (wraps HandTracker.detect, takes --banner)
from vision import hand_tracker_node_v5 as v5  # noqa: E402

if __name__ == "__main__":
    v5.DepthHandTrackerNode = v1.DemoTrackerNode
    v5.main()
