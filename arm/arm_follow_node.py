#!/usr/bin/env python3
"""
arm_follow_node.py — one xArm7 follows one wrist: gate -> clutch mapping -> differential IK.

Run one per side under its namespace (`--ros-args -r __ns:=/left`):
  in   wrist_pose        PoseStamped in camera_optical, from vision/hand_tracker_node.py
  out  arm/joint_states  JointState joint1..joint7, at --rate (simulation, RViz)

Each wrist sample goes through arm/gate.py first. A rejected sample is dropped; a
re-engage latches the wrist pose to the arm's *current* pose, so the arm never jumps
when the hand is re-found (docs/results/m1_arm_ik.md). The IK steps toward the latest
target at --rate with the joint speed capped (arm/ik_v2.py), so a far target is
approached at that speed instead of in one command.

The rest posture the IK's null space pulls toward is the joint vector the arm starts
from. A real xArm7 can sit several turns away from zero (joint1 at 242 degrees,
joint3 at -243 on the lab's left arm, 2026-10-19); pulling that towards a zero-centred
rest pose would unwind it a full turn. Starting from where it is avoids that.

Usage (with .venv-arm):
    arm/arm_follow_node.py --ros-args -r __ns:=/left
    arm/arm_follow_node.py --q0 0 -0.3 0 0.7 0 1.0 0 --max-joint-speed 1.0 --ros-args -r __ns:=/right
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm.follow import RelativeFollower  # noqa: E402
from arm.gate import Verdict, WristGate  # noqa: E402
from arm.ik_v2 import DEFAULT_Q_REST, DiffIK  # noqa: E402

JOINTS = [f"joint{i}" for i in range(1, 8)]
QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)


class ArmFollowNode(Node):
    def __init__(self, q0: np.ndarray, rate: float, max_joint_speed: float | None) -> None:
        super().__init__("arm_follow")
        self.ik = DiffIK(q_rest=q0, max_joint_speed=max_joint_speed)
        self.q = q0.copy()
        self.gate = WristGate()
        self.follower = RelativeFollower()
        self.target: tuple[np.ndarray, np.ndarray] | None = None
        self.dt = 1.0 / rate
        self.counts = {v: 0 for v in Verdict}
        self._pub = self.create_publisher(JointState, "arm/joint_states", QOS)
        self.create_subscription(PoseStamped, "wrist_pose", self._on_wrist, QOS)
        self.create_timer(self.dt, self._tick)
        self.create_timer(5.0, self._report)
        self.get_logger().info(f"start q = {np.round(q0, 3).tolist()} (also the rest posture); "
                               f"{rate:.0f} Hz, joint speed cap {self.ik.max_joint_speed.max():.2f} rad/s")

    def _on_wrist(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        o = msg.pose.orientation
        R = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        verdict = self.gate(t, p, R)
        self.counts[verdict] += 1
        if verdict is Verdict.REJECT:
            return
        if verdict is Verdict.REENGAGE:
            self.follower.engage(p, R, *self.ik.fk(self.q))
        self.target = self.follower.target(p, R)

    def _tick(self) -> None:
        if self.target is not None:
            self.q = self.ik.step(self.q, *self.target, self.dt).q_next
        msg = JointState(name=JOINTS, position=self.q.tolist())
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)

    def _report(self) -> None:
        self.get_logger().info("wrist samples " + ", ".join(f"{v.value} {n}" for v, n in self.counts.items()))


def wait_for_joints() -> np.ndarray:
    """The arm's measured joints, once: the start pose and the rest posture (see the top)."""
    node = rclpy.create_node("arm_follow_start")
    got: list[np.ndarray] = []
    node.create_subscription(JointState, "arm/measured_joint_states",
                             lambda m: got.append([dict(zip(m.name, m.position))[j] for j in JOINTS]), QOS)
    node.get_logger().info("waiting for arm/measured_joint_states")
    while rclpy.ok() and not got:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    return np.array(got[0], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    start = p.add_mutually_exclusive_group()
    start.add_argument("--q0", type=float, nargs=7, help="start joints, rad (default: the IK's rest posture)")
    start.add_argument("--q0-from-robot", action="store_true",
                       help="wait for arm/measured_joint_states (arm/xarm_bridge_node.py) and start there")
    p.add_argument("--rate", type=float, default=100.0, help="IK steps per second (default 100)")
    p.add_argument("--max-joint-speed", type=float, help="rad/s cap on every joint (default: URDF limit)")
    args = p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    q0 = np.array(args.q0) if args.q0 else DEFAULT_Q_REST.copy()
    if args.q0_from_robot:
        q0 = wait_for_joints()
    node = ArmFollowNode(q0, args.rate, args.max_joint_speed)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
