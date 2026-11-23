#!/usr/bin/env python3
"""
record_run.py — log a real-arm run as CSV on stdout, one line per message:

    <side>_wrist,t,x,y,z          /<side>/wrist_pose (camera frame, m)
    <side>_cmd,t,q1..q7           /<side>/arm/joint_states (the follow node's command, rad)
    <side>_meas,t,q1..q7          /<side>/arm/measured_joint_states (encoders, rad)
    <side>_diag,t,<text>          /<side>/wrist_depth_diag (depth fusion source etc.)

t is seconds since the recorder started. tools/run_real_ego.sh starts it; it stops on
SIGINT or SIGTERM.
"""
import signal
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

SIDES = ("left", "right")


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("record_run")
    t0 = time.monotonic()
    qos = QoSProfile(depth=50)

    def out(key, fields):
        print(f"{key},{time.monotonic() - t0:.4f}," + ",".join(fields), flush=True)

    for side in SIDES:
        node.create_subscription(
            PoseStamped, f"/{side}/wrist_pose",
            lambda m, k=f"{side}_wrist": out(k, [f"{v:.5f}" for v in (m.pose.position.x, m.pose.position.y,
                                                                     m.pose.position.z)]), qos)
        for topic, key in (("arm/joint_states", "cmd"), ("arm/measured_joint_states", "meas")):
            node.create_subscription(
                JointState, f"/{side}/{topic}",
                lambda m, k=f"{side}_{key}": out(k, [f"{v:.5f}" for v in m.position]), qos)
        node.create_subscription(String, f"/{side}/wrist_depth_diag",
                                 lambda m, k=f"{side}_diag": out(k, [m.data]), qos)

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    while not stop:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
