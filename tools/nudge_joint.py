#!/usr/bin/env python3
"""
nudge_joint.py — the first real-arm test (docs/03_arm_follow_design_v2.md, 4, R1).

Commands one joint `--delta` rad away from the arm's measured pose, holds, and brings it
back, through arm/xarm_bridge_node_v2.py (which must be running; it does the speed limit,
the excursion box and the dry-run/enable decision). This tool only publishes commands.

Usage (with .venv-arm, bridge already up under /left):
    tools/nudge_joint.py --joint 1 --delta 0.05 --ros-args -r __ns:=/left
"""
from __future__ import annotations

import argparse
import sys
import time

import rclpy
from sensor_msgs.msg import JointState

JOINTS = [f"joint{i}" for i in range(1, 8)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint", type=int, choices=range(1, 8), required=True)
    p.add_argument("--delta", type=float, default=0.05, help="rad (default 0.05, about 3 degrees)")
    p.add_argument("--hold", type=float, default=3.0, help="seconds at each end (default 3)")
    args = p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    if abs(args.delta) > 0.2:
        raise SystemExit("--delta above 0.2 rad is not a nudge")

    rclpy.init()
    node = rclpy.create_node("nudge_joint")
    measured: list[list[float]] = []
    node.create_subscription(JointState, "arm/measured_joint_states",
                             lambda m: measured.append([dict(zip(m.name, m.position))[j] for j in JOINTS]), 10)
    pub = node.create_publisher(JointState, "arm/joint_states", 10)
    while rclpy.ok() and not measured:
        rclpy.spin_once(node, timeout_sec=0.1)
    start = measured[-1]
    out = list(start)
    out[args.joint - 1] += args.delta
    log = node.get_logger()
    try:
        for label, q in (("out", out), ("back", start)):
            log.info(f"{label}: joint{args.joint} -> {q[args.joint - 1]:+.4f} rad")
            t_end = time.monotonic() + args.hold
            while time.monotonic() < t_end:
                pub.publish(JointState(name=JOINTS, position=q))
                rclpy.spin_once(node, timeout_sec=0.02)
        log.info(f"done; measured joint{args.joint} = {measured[-1][args.joint - 1]:+.4f} rad "
                 f"(start {start[args.joint - 1]:+.4f})")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
