#!/usr/bin/env python3
"""
record_run_v3.py — record_run.py plus the hands. CSV on stdout, one line per message:

    <side>_wrist,t,x,y,z          /<side>/wrist_pose (camera frame, m)
    <side>_wrist_q,t,x,y,z,w      the same message's orientation (quaternion; new in v3)
    <side>_cmd,t,q1..q7           /<side>/arm/joint_states (the follow node's command, rad)
    <side>_meas,t,q1..q7          /<side>/arm/measured_joint_states (encoders, rad)
    <side>_diag,t,<text>          /<side>/wrist_depth_diag (depth fusion source etc.)
    <side>_hand_cmd,t,q1..q20     /<side>/allegro_hand_position_controller/commands (rad, controller order:
                                  thumb joint00-03, index 10-13, middle 20-23, ring 30-33, little 40-43)
    <side>_hand_meas,t,q1..q20    /<side>/joint_states (the hand's encoders, same order)

t is seconds since the recorder started. tools/run_teleop_real.sh starts it; it stops on
SIGINT or SIGTERM.
"""
import signal
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

SIDES = ("left", "right")
HAND_JOINTS = [f"joint{f}{j}" for f in range(5) for j in range(4)]


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("record_run")  # one recorder at a time
    t0 = time.monotonic()
    qos = QoSProfile(depth=50)

    def out(key, fields):
        print(f"{key},{time.monotonic() - t0:.4f}," + ",".join(fields), flush=True)

    for side in SIDES:
        node.create_subscription(
            PoseStamped, f"/{side}/wrist_pose",
            lambda m, k=f"{side}_wrist": out(k, [f"{v:.5f}" for v in (m.pose.position.x, m.pose.position.y,
                                                                     m.pose.position.z)]), qos)
        node.create_subscription(
            PoseStamped, f"/{side}/wrist_pose",
            lambda m, k=f"{side}_wrist_q": out(k, [f"{v:.5f}" for v in (m.pose.orientation.x, m.pose.orientation.y,
                                                                       m.pose.orientation.z, m.pose.orientation.w)]), qos)
        for topic, key in (("arm/joint_states", "cmd"), ("arm/measured_joint_states", "meas")):
            node.create_subscription(
                JointState, f"/{side}/{topic}",
                lambda m, k=f"{side}_{key}": out(k, [f"{v:.5f}" for v in m.position]), qos)
        node.create_subscription(String, f"/{side}/wrist_depth_diag",
                                 lambda m, k=f"{side}_diag": out(k, [m.data]), qos)
        node.create_subscription(Float64MultiArray, f"/{side}/allegro_hand_position_controller/commands",
                                 lambda m, k=f"{side}_hand_cmd": out(k, [f"{v:.4f}" for v in m.data]), qos)

        def hand_meas(m, k=f"{side}_hand_meas"):
            q = {n.removeprefix("ah_"): p for n, p in zip(m.name, m.position)}
            if all(j in q for j in HAND_JOINTS):
                out(k, [f"{q[j]:.4f}" for j in HAND_JOINTS])
        node.create_subscription(JointState, f"/{side}/joint_states", hand_meas, qos)

    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))
    while not stop:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
