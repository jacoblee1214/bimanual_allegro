#!/usr/bin/env python3
"""
xarm_bridge_node_v2.py — the only node that talks to a real xArm7. Dry run unless --enable.

v2: v1 read get_servo_angle() with its default is_real=False, which is the controller's
target, not the encoders. The first real test (R1, 2026-10-19) logged a perfect
0.0500 rad step that way, saying nothing about how the joint followed. v2 publishes
the encoders, and the target separately. It also turns SIGINT and SIGTERM into
KeyboardInterrupt itself, instead of rclpy's handler shutting the context mid-spin,
which made v1 exit with an RCLError traceback (the arm was still stopped by the
`finally`). Installing the handlers explicitly matters: a process started in the
background inherits SIGINT as ignored, and without rclpy's handler it never stopped.

  out  arm/measured_joint_states  JointState, encoder angles (is_real=True) at --rate
  out  arm/target_joint_states    JointState, the controller's target angles
  in   arm/joint_states           JointState, the follow node's command

What stands between the command and the motors, in order:
  1. Nothing is sent without --enable. Without it the node connects, reads and
     publishes the arm's joints, and logs what it would have sent.
  2. Stale command: no command for --stale-after seconds -> nothing is sent, and the
     arm holds where it is (servo mode executes only the last instruction).
  3. Excursion box: a command with any joint more than --max-excursion rad from where it
     was when the node started is ignored as a whole, and the arm holds. (Clipping each
     joint on its own would keep the joints in the box but send the hand somewhere
     nobody asked for; the R2 dry run, 2026-10-19, pinned five joints to the box edges.)
  4. Speed: each tick moves at most --max-joint-speed * dt towards the command, all
     joints scaled together so the direction is kept.
  5. The controller's own checks (limits, collision detection) stay on.
On exit (Ctrl-C or any error) the arm is put in state 4, stop.

Only with --enable does it call motion_enable, set_mode(1) (servo) and set_state(0):
the calls that let the arm move. Read the plan in docs/ before using it.

The IP is never stored in the repository: pass --ip, or --side to read it from
config/robots.local.yaml (git-ignored).

Usage (with .venv-arm):
    arm/xarm_bridge_node_v2.py --side left --ros-args -r __ns:=/left             # dry run
    arm/xarm_bridge_node_v2.py --side left --enable --max-joint-speed 0.3 \
        --max-excursion 0.2 --ros-args -r __ns:=/left                         # moves the arm
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
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
        code, q = arm.get_servo_angle(is_radian=True, is_real=True)
        if code != 0 or arm.error_code != 0:
            raise RuntimeError(f"arm not ready: code {code}, error {arm.error_code}, warn {arm.warn_code}")
        self.q0 = np.array(q[:7], dtype=np.float64)
        self.q_sent = self.q0.copy()
        self.command: np.ndarray | None = None
        self.command_time = 0.0
        self.sent = self.held = self.clipped = self.outside = 0
        self._pub = self.create_publisher(JointState, "arm/measured_joint_states", QOS)
        self._pub_target = self.create_publisher(JointState, "arm/target_joint_states", QOS)
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

    def _publish(self, pub, is_real: bool) -> None:
        code, q = self.arm.get_servo_angle(is_radian=True, is_real=is_real)
        if code == 0:
            msg = JointState(name=JOINTS, position=[float(v) for v in q[:7]])
            msg.header.stamp = self.get_clock().now().to_msg()
            pub.publish(msg)

    def _tick(self) -> None:
        self._publish(self._pub, is_real=True)
        self._publish(self._pub_target, is_real=False)
        if self.arm.error_code != 0:
            raise RuntimeError(f"controller error {self.arm.error_code}; stopping")
        if self.command is None or time.monotonic() - self.command_time > self.stale_after:
            self.held += 1
            return

        if np.abs(self.command - self.q0).max() > self.max_excursion:
            self.outside += 1
            return
        step = self.command - self.q_sent
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
        self.get_logger().info(f"{verb} {self.sent}, held {self.held}, outside the box {self.outside}, "
                               f"speed-limited {self.clipped}; "
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
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
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
