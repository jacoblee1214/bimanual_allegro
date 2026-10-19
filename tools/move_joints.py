#!/usr/bin/env python3
"""
move_joints.py — move one xArm7 to a joint target, slowly, in the controller's position mode.

For setup moves (getting an arm out of a singular rest pose before wrist following),
not for teleoperation. The controller plans the joint-space trajectory; this tool:
  * refuses to start if the arm reports an error, or is more than --start-tolerance
    away from the --expect pose the target was computed from (it was moved since);
  * moves at --speed deg/s (default 10) with a gentle acceleration;
  * stops the arm (state 4) on Ctrl-C, SIGTERM, a controller error or a timeout.
Without --enable it prints the plan and moves nothing.

The IP comes from config/robots.local.yaml (git-ignored).

Usage (with .venv-arm):
    tools/move_joints.py --side left --expect 211.7 7.1 -243.4 50 353.6 -27.5 -191.8 \
        --target 259.1 63.9 -233.9 44.2 317.5 97.1 -314.9 [--enable]
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import yaml
from xarm.wrapper import XArmAPI

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=["left", "right"], required=True)
    p.add_argument("--expect", type=float, nargs=7, required=True, help="start pose the target assumes, deg")
    p.add_argument("--target", type=float, nargs=7, required=True, help="deg")
    p.add_argument("--speed", type=float, default=10.0, help="deg/s (default 10)")
    p.add_argument("--start-tolerance", type=float, default=2.0, help="deg (default 2)")
    p.add_argument("--enable", action="store_true", help="actually move (default: print the plan)")
    args = p.parse_args()
    if args.speed > 20.0:
        raise SystemExit("--speed above 20 deg/s is not a setup move")

    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ip = yaml.safe_load((REPO / "config" / "robots.local.yaml").read_text())["xarm"][args.side]
    arm = XArmAPI(ip, is_radian=False)
    moving = False
    try:
        if arm.error_code != 0:
            raise SystemExit(f"arm reports error {arm.error_code}; clear it first")
        code, q = arm.get_servo_angle(is_radian=False, is_real=True)
        q = np.array(q[:7])
        off = np.abs(q - np.array(args.expect)).max()
        if code != 0 or off > args.start_tolerance:
            raise SystemExit(f"arm is {off:.1f} deg from --expect (now {np.round(q, 1).tolist()}); recompute the target")
        travel = np.array(args.target) - q
        duration = np.abs(travel).max() / args.speed
        print(f"{args.side}: now    {np.round(q, 1).tolist()}")
        print(f"{args.side}: target {np.round(args.target, 1).tolist()}")
        print(f"{args.side}: travel {np.round(travel, 1).tolist()}, about {duration:.0f} s at {args.speed:.0f} deg/s")
        if not args.enable:
            print("dry run: nothing moved")
            return

        for call, ret in (("motion_enable", arm.motion_enable(True)), ("set_mode(0)", arm.set_mode(0)),
                          ("set_state(0)", arm.set_state(0))):
            if ret != 0:
                raise RuntimeError(f"{call} returned {ret}")
        time.sleep(0.5)
        moving = True
        ret = arm.set_servo_angle(angle=list(args.target), speed=args.speed, mvacc=4 * args.speed,
                                  is_radian=False, wait=False)
        if ret != 0:
            raise RuntimeError(f"set_servo_angle returned {ret}")
        deadline = time.monotonic() + 2 * duration + 10
        time.sleep(0.5)
        while arm.state == 1:                       # 1 = in motion
            if arm.error_code != 0:
                raise RuntimeError(f"controller error {arm.error_code} during the move")
            if time.monotonic() > deadline:
                raise RuntimeError("move took twice as long as planned")
            time.sleep(0.05)
        moving = False
        _, q = arm.get_servo_angle(is_radian=False, is_real=True)
        print(f"{args.side}: done   {np.round(q[:7], 1).tolist()} "
              f"(max error {np.abs(np.array(q[:7]) - np.array(args.target)).max():.2f} deg), error {arm.error_code}")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if moving:
            arm.set_state(4)
            print("stopped (state 4)")
        arm.disconnect()


if __name__ == "__main__":
    main()
