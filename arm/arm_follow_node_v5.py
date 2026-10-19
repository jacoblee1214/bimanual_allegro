#!/usr/bin/env python3
"""
arm_follow_node_v5.py — one xArm7 follows one wrist: gate -> clutch mapping -> differential IK.

Run one per side under its namespace (`--ros-args -r __ns:=/left`):
  in   wrist_pose               PoseStamped in camera_optical, from vision/hand_tracker_node.py
  in   arm/measured_joint_states  (with --q0-from-robot) the start pose, once
  out  arm/joint_states         JointState joint1..joint7, at --rate

Each wrist sample goes through arm/gate.py; a re-engage latches the wrist to the arm's
current pose so the arm never jumps. The IK (arm/ik_v2.py) steps towards the latest
target at --rate with the joint speed capped. The rest posture is the start pose, so a
real arm wound several turns is not unwound.

Options added along the way, from the real tests (docs/results/r2_*, r3_*):
  --position-only  (v2) keep the engaged EEF orientation; the vision's wrist orientation
                   still jumps.
  --scale          (v3) EEF motion per unit of hand motion; a hand moves faster than the arm.
  --radius         (v3) keep the EEF target within this of the start.
  --joint-box      (v4) keep every joint within this of the start, inside the bridge's box.

v4: the --radius sphere is centred on the EEF pose the node *starts* from, and
--joint-box keeps the IK itself inside the bridge's excursion box. In the v3 retry of R3
the hand left the image often (356 of 893 frames found), every re-clutch moved the
sphere's centre to where the arm then was, the drift added up to a 157 mm command, and
the IK walked joints 0.58 rad out, past the bridge's 0.3 rad box, so the bridge ignored
it and the arm stood still. Now neither can leave: the sphere stays put and the joints
are clipped 0.02 rad inside the box, so at the edge the arm slides instead of stopping.

v5: --camera reads the camera -> robot rotation (R_CB) from a file instead of assuming
the operator faces the camera. For the egocentric RealSense, tools/measure_camera_tilt.py
writes config/ego_camera.yaml from the IMU, so "forward" is the camera's levelled viewing
direction and "up" is against gravity, whatever the camera's tilt.

Usage (with .venv-arm, bridge up under /left):
    arm/arm_follow_node_v5.py --q0-from-robot --position-only --scale 0.5 --radius 0.07 \
        --joint-box 0.3 --max-joint-speed 0.3 --camera config/ego_camera.yaml --ros-args -r __ns:=/left
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
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
    def __init__(self, q0: np.ndarray, rate: float, max_joint_speed: float | None,
                 position_only: bool = False, radius: float | None = None, scale: float = 1.0,
                 joint_box: float | None = None, R_cb: np.ndarray | None = None) -> None:
        super().__init__("arm_follow")
        self.ik = DiffIK(q_rest=q0, max_joint_speed=max_joint_speed)
        self.q = q0.copy()
        self.gate = WristGate()
        self.follower = RelativeFollower(scale=scale) if R_cb is None else RelativeFollower(R_cb=R_cb, scale=scale)
        self.radius = radius
        self._p_start = self.ik.fk(q0)[0]
        if joint_box is not None:
            self.ik.q_lo = np.maximum(self.ik.q_lo, q0 - joint_box + 0.02)
            self.ik.q_hi = np.minimum(self.ik.q_hi, q0 + joint_box - 0.02)
        self.target: tuple[np.ndarray, np.ndarray] | None = None
        self.dt = 1.0 / rate
        self.position_only = position_only
        self._R_engaged: np.ndarray | None = None
        self.counts = {v: 0 for v in Verdict}
        self._pub = self.create_publisher(JointState, "arm/joint_states", QOS)
        self.create_subscription(PoseStamped, "wrist_pose", self._on_wrist, QOS)
        self.create_timer(self.dt, self._tick)
        self.create_timer(5.0, self._report)
        self.get_logger().info(f"start q = {np.round(q0, 3).tolist()} (also the rest posture); "
                               f"{rate:.0f} Hz, joint speed cap {self.ik.max_joint_speed.max():.2f} rad/s"
                               + ("; position only" if position_only else "")
                               + (f"; within {radius * 100:.0f} cm" if radius is not None else "")
                               + f"; scale {scale:.2f}"
                               + (f"; joints within {joint_box:.2f} rad of start" if joint_box is not None else ""))

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
            p_eef, R_eef = self.ik.fk(self.q)
            self.follower.engage(p, R, p_eef, R_eef)
            self._R_engaged = R_eef
        p_target, R_target = self.follower.target(p, R)
        if self.radius is not None:
            offset = p_target - self._p_start
            norm = np.linalg.norm(offset)
            if norm > self.radius:
                p_target = self._p_start + offset * (self.radius / norm)
        self.target = (p_target, self._R_engaged if self.position_only else R_target)

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
                       help="wait for arm/measured_joint_states (arm/xarm_bridge_node_v3.py) and start there")
    p.add_argument("--rate", type=float, default=100.0, help="IK steps per second (default 100)")
    p.add_argument("--max-joint-speed", type=float, help="rad/s cap on every joint (default: URDF limit)")
    p.add_argument("--position-only", action="store_true", help="keep the engaged EEF orientation")
    p.add_argument("--radius", type=float, help="m; keep the EEF target within this of the engaged position")
    p.add_argument("--scale", type=float, default=1.0, help="EEF motion per unit of hand motion (default 1.0)")
    p.add_argument("--joint-box", type=float, help="rad; keep every joint this close to the start (match the bridge)")
    p.add_argument("--camera", type=Path, help="YAML with R_CB (tools/measure_camera_tilt.py); default: facing the operator")
    args = p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    q0 = np.array(args.q0) if args.q0 else DEFAULT_Q_REST.copy()
    if args.q0_from_robot:
        q0 = wait_for_joints()
    R_cb = np.array(yaml.safe_load(args.camera.read_text())["R_CB"]) if args.camera else None
    node = ArmFollowNode(q0, args.rate, args.max_joint_speed, args.position_only, args.radius, args.scale,
                         args.joint_box, R_cb)
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
