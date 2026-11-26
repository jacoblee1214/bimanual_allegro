#!/usr/bin/env python3
"""
arm_follow_node_v7.py — arm_follow_node_v6.py plus a sideways limit and a rotation limit on the EEF target.

With the arms started turned inwards (config/teleop_center.yaml, docs/12_same_motion.md) the
two hands are 0.4 m apart and each may move 0.2 m: without a limit they can meet. Each arm
runs its own follow node and does not know the other, so each keeps to its own side:

  --y-range LO HI   m, in the arm's base frame; the EEF target's y is clipped to this
                    after the --radius sphere.
  --max-rotation D  degrees; without --position-only, the EEF target is turned at most this
                    far from the orientation the node started with (docs/13_wrist_rotation.md).
                    With it the rotation is also engaged late: after a (re-)engage the EEF keeps
                    its orientation until the wrist's orientation has moved less than 5 degrees
                    per frame for 10 frames in a row, and only then is that orientation latched
                    as the reference. The first poses after a hand appears are often PnP's, and
                    a reference latched on one of them leaves the EEF at the limit for good.

Everything else, topics included, is v6.

Usage (with .venv-arm): as v6, e.g.
    arm/arm_follow_node_v7.py --q0-from-robot --position-only --radius 0.2 --joint-box 0.9 \
        --y-range -0.38 -0.11 --camera config/ego_camera.yaml --ros-args -r __ns:=/left
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import arm_follow_node_v6 as v6  # noqa: E402


def clip_y(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.array(p, dtype=np.float64)
    out[1] = min(max(out[1], lo), hi)
    return out


SETTLE_FRAMES, SETTLE_STEP = 10, np.radians(5.0)


def limit_rotation(R: np.ndarray, R_start: np.ndarray, max_rad: float) -> np.ndarray:
    """R, or the rotation on the way from R_start to R that is max_rad from R_start."""
    rel = Rotation.from_matrix(R @ R_start.T).as_rotvec()
    angle = float(np.linalg.norm(rel))
    if angle <= max_rad:
        return R
    return Rotation.from_rotvec(rel * (max_rad / angle)).as_matrix() @ R_start


def main() -> None:
    max_rotation = None
    if "--max-rotation" in sys.argv:
        i = sys.argv.index("--max-rotation")
        max_rotation = np.radians(float(sys.argv[i + 1]))
        del sys.argv[i:i + 2]
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
            self._R_start = self.ik.fk(self.q)[1]
            self._calm, self._R_prev = 0, np.eye(3)
            if max_rotation is not None and not self.position_only:
                self.get_logger().info(f"EEF turned at most {np.degrees(max_rotation):.0f} deg from the start")
            if y_range is not None:
                y0 = self._p_start[1]
                if not y_range[0] <= y0 <= y_range[1]:
                    raise RuntimeError(f"the start (y {y0:.3f}) is outside --y-range {y_range}")
                self.get_logger().info(f"EEF y kept within [{y_range[0]:.3f}, {y_range[1]:.3f}] m (start {y0:.3f})")

        def _on_wrist(self, msg) -> None:
            engaged = self.counts[v6.Verdict.REENGAGE]
            super()._on_wrist(msg)
            if max_rotation is not None and not self.position_only and self.target is not None:
                o = msg.pose.orientation
                R = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
                if self.counts[v6.Verdict.REENGAGE] != engaged:
                    self._calm, self._R_prev = 0, R
                if self._calm < SETTLE_FRAMES:
                    step = np.linalg.norm(Rotation.from_matrix(R @ self._R_prev.T).as_rotvec())
                    self._calm = self._calm + 1 if step < SETTLE_STEP else 0
                    self._R_prev = R
                    R_eef = self.ik.fk(self.q)[1]
                    if self._calm == SETTLE_FRAMES:      # latch the rotation reference now
                        self.follower._wrist_R0, self.follower._eef_R0 = R, R_eef
                    self.target = (self.target[0], R_eef)
            if y_range is not None and self.target is not None:
                self.target = (clip_y(self.target[0], *y_range), self.target[1])
            if max_rotation is not None and not self.position_only and self.target is not None:
                self.target = (self.target[0], limit_rotation(self.target[1], self._R_start, max_rotation))

    v6.ArmFollowNode = ArmFollowNode
    v6.main()


if __name__ == "__main__":
    main()
