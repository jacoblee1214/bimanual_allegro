#!/usr/bin/env python3
"""
hand_bridge_node.py — allegro_vision_teleop/sim_bridge_node.py with a speed cap and a measured start.

The hand repository's bridge smooths the retargeted joints with an EMA and sends them at 100 Hz.
Replaying data/clips/ego_depth_work through it (2026-11-26, docs/10_real_teleop_arms_hands.md)
gave commands of up to 42 rad/s: when the tracker loses a hand and finds it again, the target
jumps and the EMA passes a quarter of the jump on in one tick. With the hands on the arms,
next to each other and above a table, that is too fast. This node adds two things and
changes nothing else (same topics, limits, coordinates and EMA):

  1. Speed cap: each joint moves at most --max-joint-speed * dt per tick (default 3 rad/s;
     normal finger motion in the same clip is below 3.6 rad/s 95 % of the time).
  2. Measured start: nothing is sent before the first joint_states message, and the filter
     starts from the measured joints instead of from the open hand.

Usage (with the hand repository's .venv; launch/teleop_real.launch.py starts it):
    hand/hand_bridge_node.py --hand left --mode real --hand-repo ~/allegro_vision_teleop \
        --ros-args -r __ns:=/left
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def limit_speed(q_sent: np.ndarray, q_wanted: np.ndarray, max_step: float) -> np.ndarray:
    """Move each joint from `q_sent` towards `q_wanted` by at most `max_step`."""
    return q_sent + np.clip(q_wanted - q_sent, -max_step, max_step)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hand", choices=["left", "right"], required=True)
    p.add_argument("--mode", choices=["real", "sim"], default="real")
    p.add_argument("--alpha", type=float, default=0.25, help="EMA factor per tick (default 0.25)")
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--max-joint-speed", type=float, default=3.0, help="rad/s, every joint (default 3)")
    p.add_argument("--left-mcp-offset", type=float, default=0.0, help="see sim_bridge_node.py; 0 for the lab's hand")
    p.add_argument("--hand-repo", type=Path,
                   default=Path(os.environ.get("ALLEGRO_TELEOP_DIR", "~/allegro_vision_teleop")).expanduser())
    args, _ = p.parse_known_args()

    sys.path.insert(0, str(args.hand_repo))
    import sim_bridge_node as base  # noqa: E402  (the hand repository's bridge)
    from safety_utils import CONTROLLER_JOINT_ORDER, N_JOINTS  # noqa: E402

    class HandBridgeNode(base.SimBridgeNode):
        def __init__(self) -> None:
            super().__init__(alpha=args.alpha, rate=args.rate, mode=args.mode, hand_side=args.hand,
                             left_mcp_offset=args.left_mcp_offset)
            self._max_step = args.max_joint_speed / args.rate
            self._sent: np.ndarray | None = None
            self._latest_target = None      # hold the measured pose until the first target
            self._filtered_angles = None
            self.get_logger().warn(f"speed cap {args.max_joint_speed:.1f} rad/s; waiting for joint_states")

        def _joint_states_callback(self, msg: JointState) -> None:
            super()._joint_states_callback(msg)     # republishes msg in URDF coordinates
            if self._sent is None:
                q = {n.removeprefix("ah_"): v for n, v in zip(msg.name, msg.position)}
                if all(j in q for j in CONTROLLER_JOINT_ORDER):
                    self._sent = np.array([q[j] for j in CONTROLLER_JOINT_ORDER], dtype=np.float64)
                    self._filtered_angles = self._sent.copy()
                    self.get_logger().warn("start from the measured joints: "
                                           f"{np.round(self._sent, 2).tolist()}")

        def _control_loop_tick(self) -> None:
            if self._sent is None:
                return
            if self._latest_target is not None:
                self._filtered_angles = (self.alpha * self._latest_target
                                         + (1.0 - self.alpha) * self._filtered_angles)
            self._sent = limit_speed(self._sent, self._filtered_angles, self._max_step)
            cmd = self._sent.copy()
            if self.mode == "real" and self.hand_side == "left" and self.left_mcp_offset:
                cmd[list(base.LEFT_MCP_JOINT_INDICES)] += self.left_mcp_offset
            self._pub.publish(Float64MultiArray(data=cmd.tolist()))
            self._seq += 1

    assert N_JOINTS == 20
    rclpy.init()
    node = HandBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
