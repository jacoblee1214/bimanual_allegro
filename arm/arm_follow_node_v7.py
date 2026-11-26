#!/usr/bin/env python3
"""
arm_follow_node_v7.py — arm_follow_node_v6.py plus a sideways limit on the EEF target.

With the arms started turned inwards (config/teleop_center.yaml, docs/12_same_motion.md) the
two hands are 0.4 m apart and each may move 0.2 m: without a limit they can meet. Each arm
runs its own follow node and does not know the other, so each keeps to its own side:

  --y-range LO HI   m, in the arm's base frame; the EEF target's y is clipped to this
                    after the --radius sphere.

Everything else, topics included, is v6.

Usage (with .venv-arm): as v6, e.g.
    arm/arm_follow_node_v7.py --q0-from-robot --position-only --radius 0.2 --joint-box 0.9 \
        --y-range -0.38 -0.11 --camera config/ego_camera.yaml --ros-args -r __ns:=/left
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import arm_follow_node_v6 as v6  # noqa: E402


def clip_y(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.array(p, dtype=np.float64)
    out[1] = min(max(out[1], lo), hi)
    return out


def main() -> None:
    y_range = None
    if "--y-range" in sys.argv:
        i = sys.argv.index("--y-range")
        y_range = (float(sys.argv[i + 1]), float(sys.argv[i + 2]))
        del sys.argv[i:i + 3]
        if not y_range[0] < y_range[1]:
            sys.exit("--y-range LO HI needs LO < HI")

    class ArmFollowNode(v6.ArmFollowNode):
        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            if y_range is not None:
                y0 = self._p_start[1]
                if not y_range[0] <= y0 <= y_range[1]:
                    raise RuntimeError(f"the start (y {y0:.3f}) is outside --y-range {y_range}")
                self.get_logger().info(f"EEF y kept within [{y_range[0]:.3f}, {y_range[1]:.3f}] m (start {y0:.3f})")

        def _on_wrist(self, msg) -> None:
            super()._on_wrist(msg)
            if y_range is not None and self.target is not None:
                self.target = (clip_y(self.target[0], *y_range), self.target[1])

    v6.ArmFollowNode = ArmFollowNode
    v6.main()


if __name__ == "__main__":
    main()
