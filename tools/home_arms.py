#!/usr/bin/env python3
"""
home_arms.py — bring both xArm7 arms to a config's start pose (home_q; default config/teleop_real.yaml).

After a teleop run the arms stand where the run ended, and tools/run_teleop_real_v2.sh only
starts from the home pose. This reads each arm (tools/check_pose_v2.py) and, with --enable,
calls tools/move_joints.py from the pose it just read to the home pose: 10 deg/s, one arm
after the other, left first. It refuses if an arm reports an error or is more than
--max-travel degrees from home in any joint (default 90): that is not "a run just ended".
Without --enable it prints the plan and moves nothing.

Usage (with .venv-arm):
    tools/home_arms.py [--config config/teleop_center.yaml] [--enable]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enable", action="store_true", help="actually move (default: print the plan)")
    p.add_argument("--max-travel", type=float, default=90.0, help="deg; refuse beyond this (default 90)")
    p.add_argument("--config", type=Path, default=REPO / "config" / "teleop_real.yaml", help="YAML with home_q")
    args = p.parse_args()
    home = yaml.safe_load(args.config.read_text())["home_q"]
    text = subprocess.run([sys.executable, str(REPO / "tools" / "check_pose_v2.py")],
                          capture_output=True, text=True).stdout
    plan = []
    for side in ("left", "right"):
        m = re.search(side + r": state (\d+), error (\d+).*?joints deg\s+\[([^\]]*)\]", text, re.S)
        if not m or m.group(2) != "0":
            sys.exit(f"{side}: not read, or the arm reports an error; not moving anything")
        now = [float(v) for v in m.group(3).split(",")]
        target = np.degrees(home[side]).round(1).tolist()
        travel = float(np.abs(np.subtract(target, now)).max())
        print(f"{side}: now {now} -> home {target}, largest joint travel {travel:.1f} deg")
        if travel > args.max_travel:
            sys.exit(f"{side}: more than {args.max_travel:.0f} deg from home; move it by hand with tools/move_joints.py")
        if travel > 0.2:
            plan.append((side, now, target))
    if not args.enable:
        print("plan only; --enable moves the arms")
        return
    for side, now, target in plan:
        cmd = [sys.executable, str(REPO / "tools" / "move_joints.py"), "--side", side,
               "--expect", *map(str, now), "--target", *map(str, target), "--enable"]
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"{side}: the move failed; stopping")
    print("both arms at home" if plan else "both arms were already at home")


if __name__ == "__main__":
    main()
