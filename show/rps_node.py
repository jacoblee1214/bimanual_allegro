#!/usr/bin/env python3
"""
rps_node.py — rock-paper-scissors between a visitor and the robot, on top of the teleop.

The node sits between the teleop and the bridges, for both sides:

    retargeting  /<side>/allegro/teleop_target_joints -> [this node] -> /<side>/allegro/target_joints -> hand bridge
    arm follow   /<side>/arm/teleop_joint_states      -> [this node] -> /<side>/arm/joint_states      -> arm bridge

While nobody plays it passes every message on unchanged (the robot mirrors the visitor). A
round starts when a hand shows a V for --trigger-hold seconds, on a message on /show/start,
or with the G key in the window. During a round the arms hold where they were, the right
arm pumps its elbow three times (--pump rad, towards the start pose), the right hand throws
and the left hand rests. Afterwards the teleop is passed on again. The bridges' speed caps
and joint box stay in force throughout: this node only chooses targets.

  --mode fair     the robot's throw is drawn before the count
  --mode reader   the robot reads the visitor's opening fingers and throws what beats them
                  (M toggles the mode in the window)

/show/state (std_msgs/String, JSON, 30 Hz): phase, label, robot, human, winner, react_ms, score.

Usage (with .venv-vision; launch/teleop_real_v2.launch.py show:=rps starts it):
    show/rps_node.py --mode fair [--no-window] [--auto-start 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32MultiArray, Float64MultiArray, String

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from show import rps  # noqa: E402

SIDES = ("left", "right")
PLAYER = "right"            # the robot's throwing hand and pumping arm
ELBOW = "joint4"
QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)


def pump_offset(q_hold: float, q_start: float, amplitude: float, box: float) -> float:
    """The elbow's pump stroke, rad: towards the start pose, and never out of the joint box."""
    sign = -1.0 if q_hold >= q_start else 1.0
    room = box - abs(q_hold + sign * amplitude - q_start)
    return sign * amplitude if room >= 0.0 else 0.0


class RpsNode(Node):
    def __init__(self, args) -> None:
        super().__init__("rps_show")
        self.args = args
        self.game = rps.Game(np.random.default_rng(), mode=args.mode)
        self.ext: dict[str, tuple[float, np.ndarray]] = {}
        self.arm_start: dict[str, dict] = {}
        self.arm_last: dict[str, JointState] = {}
        self.arm_hold: dict[str, dict] = {}
        self.vee_since: dict[str, float | None] = {s: None for s in SIDES}
        self.hand_side = PLAYER             # the visitor's hand being read
        self.idle_since = time.monotonic()
        self.first_hand: float | None = None
        self.shown: str | None = None
        self.stroke = 0.0
        self.was_playing = False
        self.hand_pub = {s: self.create_publisher(Float64MultiArray, f"/{s}/allegro/target_joints", QOS) for s in SIDES}
        self.arm_pub = {s: self.create_publisher(JointState, f"/{s}/arm/joint_states", QOS) for s in SIDES}
        self.state_pub = self.create_publisher(String, "/show/state", QOS)
        for s in SIDES:
            self.create_subscription(Float64MultiArray, f"/{s}/allegro/teleop_target_joints",
                                     lambda m, s=s: self._on_hand(s, m), QOS)
            self.create_subscription(JointState, f"/{s}/arm/teleop_joint_states",
                                     lambda m, s=s: self._on_arm(s, m), QOS)
            self.create_subscription(Float32MultiArray, f"/{s}/allegro/vision/world_landmarks",
                                     lambda m, s=s: self._on_landmarks(s, m), QOS)
        self.create_subscription(Empty, "/show/start", lambda _: self._start("topic"), QOS)
        self.create_timer(0.01, self._tick)
        self.create_timer(1.0 / 30.0, self._show)
        self.get_logger().info(f"rock-paper-scissors: mode {args.mode}; show a V for {args.trigger_hold:.1f} s to play")

    @property
    def playing(self) -> bool:
        return self.game.phase != "mirror"

    def _on_hand(self, side: str, msg: Float64MultiArray) -> None:
        if not self.playing:
            self.hand_pub[side].publish(msg)

    def _on_arm(self, side: str, msg: JointState) -> None:
        self.arm_start.setdefault(side, dict(zip(msg.name, msg.position)))
        self.arm_last[side] = msg
        if not self.playing:
            self.arm_pub[side].publish(msg)

    def _on_landmarks(self, side: str, msg: Float32MultiArray) -> None:
        if len(msg.data) != 63:
            return
        now = time.monotonic()
        ext = rps.extension(msg.data)
        self.ext[side] = (now, ext)
        if self.first_hand is None:
            self.first_hand = now
        if self.playing:
            if side == self.hand_side:
                self.game.step(now, ext)
            return
        if rps.classify(ext) == rps.SCISSORS and now - self.idle_since > self.args.cooldown:
            self.vee_since[side] = self.vee_since[side] or now
            if now - self.vee_since[side] >= self.args.trigger_hold:
                self._start(f"V sign, {side} hand", side)
        else:
            self.vee_since[side] = None

    def _start(self, why: str, hand_side: str | None = None) -> None:
        if self.playing:
            return
        now = time.monotonic()
        fresh = [s for s in SIDES if s in self.ext and now - self.ext[s][0] < 0.3]
        self.hand_side = hand_side or (fresh[0] if fresh else PLAYER)
        self.arm_hold = {s: dict(zip(m.name, m.position)) for s, m in self.arm_last.items()}
        self.stroke = 0.0
        if PLAYER in self.arm_hold:
            self.stroke = pump_offset(self.arm_hold[PLAYER][ELBOW], self.arm_start[PLAYER][ELBOW],
                                      self.args.pump, self.args.joint_box)
        self.vee_since = {s: None for s in SIDES}
        self.game.start(now)
        self.get_logger().info(f"round starts ({why}); reading the visitor's {self.hand_side} hand; "
                               f"elbow stroke {self.stroke:+.3f} rad")

    def _tick(self) -> None:
        now = time.monotonic()
        seen = self.ext.get(self.hand_side)
        if self.playing and (seen is None or now - seen[0] > 0.2):
            self.game.step(now, None)                   # no fresh hand: keep the round's clock running
        if not self.playing:
            if self.was_playing:                        # the round has just ended
                self.was_playing, self.idle_since, g = False, now, self.game
                self.get_logger().info(f"round over: robot {self.shown}, visitor {g.human}, winner {g.winner}, "
                                       f"react {g.react_ms} ms, score {g.score}")
            elif (self.args.auto_start is not None and self.first_hand is not None
                    and now - self.first_hand >= self.args.auto_start):
                self.args.auto_start = None
                self._start("auto start")
            return
        self.was_playing = True
        self.shown = self.game.robot
        cheer = self.game.phase == "result" and self.game.winner == "robot" and int(now * 4.0) % 2 == 0
        targets = {PLAYER: rps.POSES[self.game.robot],
                   "left" if PLAYER == "right" else "right": rps.POSES[rps.PAPER if cheer else "relaxed"]}
        for side, q in targets.items():
            self.hand_pub[side].publish(Float64MultiArray(data=q.tolist()))
        for side, hold in self.arm_hold.items():
            q = dict(hold)
            if side == PLAYER:
                q[ELBOW] += self.stroke * self.game.pump(now)
            msg = JointState(name=list(q), position=[float(v) for v in q.values()])
            msg.header.stamp = self.get_clock().now().to_msg()
            self.arm_pub[side].publish(msg)

    def _show(self) -> None:
        g = self.game
        state = {"phase": g.phase, "label": g.label, "mode": g.mode, "robot": g.robot, "human": g.human,
                 "winner": g.winner, "react_ms": g.react_ms, "score": g.score}
        self.state_pub.publish(String(data=json.dumps(state)))
        if self.args.no_window:
            return
        import cv2
        img = np.zeros((480, 960, 3), np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX

        def text(s, y, scale, colour=(255, 255, 255)):
            w = cv2.getTextSize(s, font, scale, 2)[0][0]
            cv2.putText(img, s, ((960 - w) // 2, y), font, scale, colour, 2, cv2.LINE_AA)

        if g.phase == "mirror":
            text("MOVE YOUR HANDS - THE ROBOT FOLLOWS", 170, 1.2)
            text("Show a V sign to play rock-paper-scissors", 250, 1.0, (120, 220, 255))
        else:
            text(g.label, 190, 3.0, (120, 220, 255) if g.phase != "result" else (120, 255, 160))
            if g.phase == "result":
                text(f"robot: {(self.shown or '').upper()}    you: {(g.human or '-').upper()}", 280, 1.2)
                if g.react_ms is not None:
                    text(f"the robot read your hand in {g.react_ms:.0f} ms", 340, 0.9, (180, 180, 180))
        text(f"ROBOT {g.score['robot']}  :  {g.score['you']} YOU     draws {g.score['draw']}", 430, 1.0)
        cv2.putText(img, f"mode: {g.mode}  (G: play, M: mode)", (12, 24), font, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.imshow("rock-paper-scissors", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("g"):
            self._start("key")
        elif key == ord("m") and not self.playing:
            g.mode = "reader" if g.mode == "fair" else "fair"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["fair", "reader"], default="fair")
    p.add_argument("--trigger-hold", type=float, default=1.0, help="s a V sign must be held to start (default 1)")
    p.add_argument("--cooldown", type=float, default=3.0, help="s after a round before a V sign counts again")
    p.add_argument("--pump", type=float, default=0.08, help="rad, the elbow's stroke during the count (default 0.08)")
    p.add_argument("--joint-box", type=float, default=0.3, help="rad, the bridge's box around the start pose")
    p.add_argument("--auto-start", type=float, help="start one round this many s after the first hand (tests)")
    p.add_argument("--no-window", action="store_true")
    args, _ = p.parse_known_args()
    rclpy.init()
    node = RpsNode(args)
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
