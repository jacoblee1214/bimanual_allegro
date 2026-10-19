#!/usr/bin/env python3
"""
xarm_bridge_node.py — the only node that talks to a real xArm7. Dry run unless --enable.

  out  arm/measured_joint_states  JointState, read from the arm at --rate
  in   arm/joint_states           JointState, the follow node's command

What stands between the command and the motors, in order:
  1. Nothing is sent without --enable. Without it the node connects, reads and
     publishes the arm's joints, and logs what it would have sent.
  2. Stale command: no command for --stale-after seconds -> nothing is sent, and the
     arm holds where it is (servo mode executes only the last instruction).
  3. Excursion box: every joint stays within --max-excursion rad of where it was when
     the node started. The first real tests are meant to be small.
  4. Speed: each tick moves at most --max-joint-speed * dt towards the command, all
     joints scaled together so the direction is kept.
  5. The controller's own checks (limits, collision detection) stay on.
On exit (Ctrl-C or any error) the arm is put in state 4, stop.

Only with --enable does it call motion_enable, set_mode(1) (servo) and set_state(0):
the calls that let the arm move. Read the plan in docs/ before using it.

The IP is never stored in the repository: pass --ip, or --side to read it from
config/robots.local.yaml (git-ignored).

Usage (with .venv-arm):
    arm/xarm_bridge_node.py --side left --ros-args -r __ns:=/left             # dry run
    arm/xarm_bridge_node.py --side left --enable --max-joint-speed 0.3 \
        --max-excursion 0.2 --ros-args -r __ns:=/left                         # moves the arm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from xarm.wrapper import XArmAPI

REPO = Path(__file__).resolve().parents[1]
JOINTS = [f"joint{i}" for i in range(1, 8)]
QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)


def robot_ip(side: str) -> str:
    return yaml.safe_load((REPO / "config" / "robots.local.yaml").read_text())["xarm"][side]


class XArmBridgeNode(Node):
    def __init__(self, arm: XArmAPI, enable: bool, rate: float, max_joint_speed: float,
                 max_excursion: float, stale_after: float) -> None:
        super().__init__("xarm_bridge")
        self.arm, self.enable = arm, enable
        self.dt, self.max_step = 1.0 / rate, max_joint_speed / rate
        self.max_excursion, self.stale_after = max_excursion, stale_after
        code, q = arm.get_servo_angle(is_radian=True)
        if code != 0 or arm.error_code != 0:
            raise RuntimeError(f"arm not ready: code {code}, error {arm.error_code}, warn {arm.warn_code}")
        self.q0 = np.array(q[:7], dtype=np.float64)
        self.q_sent = self.q0.copy()
        self.command: np.ndarray | None = None
        self.command_time = 0.0
        self.sent = self.held = self.clipped = 0
        self._pub = self.create_publisher(JointState, "arm/measured_joint_states", QOS)
        self.create_subscription(JointState, "arm/joint_states", self._on_command, QOS)
        self.create_timer(self.dt, self._tick)
        self.create_timer(5.0, self._report)
        self.get_logger().info(
            f"{'ENABLED' if enable else 'dry run'}: start q = {np.round(self.q0, 3).tolist()}, "
            f"{rate:.0f} Hz, <= {max_joint_speed:.2f} rad/s, within {max_excursion:.2f} rad of start, "
            f"stale after {stale_after:.2f} s")

    def _on_command(self, msg: JointState) -> None:
        q = dict(zip(msg.name, msg.position))
        self.command = np.array([q[j] for j in JOINTS])
        self.command_time = time.monotonic()

    def _tick(self) -> None:
        code, q = self.arm.get_servo_angle(is_radian=True)
        if code == 0:
            msg = JointState(name=JOINTS, position=[float(v) for v in q[:7]])
            msg.header.stamp = self.get_clock().now().to_msg()
            self._pub.publish(msg)
        if self.arm.error_code != 0:
            raise RuntimeError(f"controller error {self.arm.error_code}; stopping")
        if self.command is None or time.monotonic() - self.command_time > self.stale_after:
            self.held += 1
            return

        target = np.clip(self.command, self.q0 - self.max_excursion, self.q0 + self.max_excursion)
        step = target - self.q_sent
        biggest = np.abs(step).max()
        if biggest > self.max_step:
            step *= self.max_step / biggest
            self.clipped += 1
        self.q_sent = self.q_sent + step
        if self.enable:
            ret = self.arm.set_servo_angle_j(self.q_sent.tolist(), is_radian=True)
            if ret != 0:
                raise RuntimeError(f"set_servo_angle_j returned {ret}; stopping")
        self.sent += 1

    def _report(self) -> None:
        verb = "sent" if self.enable else "would have sent"
        self.get_logger().info(f"{verb} {self.sent}, held {self.held}, speed-limited {self.clipped}; "
                               f"offset from start {np.round(self.q_sent - self.q0, 3).tolist()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--ip")
    where.add_argument("--side", choices=["left", "right"], help="read the IP from config/robots.local.yaml")
    p.add_argument("--enable", action="store_true", help="actually move the arm (default: dry run)")
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--max-joint-speed", type=float, default=0.3, help="rad/s, every joint (default 0.3)")
    p.add_argument("--max-excursion", type=float, default=0.2, help="rad from the start pose (default 0.2)")
    p.add_argument("--stale-after", type=float, default=0.2, help="s without a command before holding")
    args = p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    arm = XArmAPI(args.ip or robot_ip(args.side), is_radian=True)
    if not arm.connected:
        raise SystemExit("could not connect")
    rclpy.init()
    node = None
    try:
        if args.enable:
            for call, ret in (("motion_enable", arm.motion_enable(True)), ("set_mode(1)", arm.set_mode(1)),
                              ("set_state(0)", arm.set_state(0))):
                if ret != 0:
                    raise RuntimeError(f"{call} returned {ret}")
            time.sleep(0.5)  # the controller needs a moment before servo commands
        node = XArmBridgeNode(arm, args.enable, args.rate, args.max_joint_speed,
                              args.max_excursion, args.stale_after)
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if args.enable:
            arm.set_state(4)
        arm.disconnect()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
