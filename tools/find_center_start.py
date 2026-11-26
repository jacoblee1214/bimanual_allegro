#!/usr/bin/env python3
"""
find_center_start.py — a start pose with the two hands as far apart as the operator's.

From the home pose each arm reaches straight ahead, so the robot's hands are as far apart as
the bases (0.98 m between the base cables, user, 2026-11-26) while the operator's are 0.3 to
0.4 m apart: the robot's hands can never meet (docs/12_same_motion.md). Here each arm starts
turned inwards instead: the flange at (x, -/+(base distance - spacing)/2, z) in its own base
frame, with the home pose's flange orientation, found by the follow node's IK from the home
pose. For every candidate the reach test of tools/eval_reach_from_home.py is repeated.

Usage (with .venv-arm):
    tools/find_center_start.py [--bases 0.98] [--spacing 0.3 0.4] [--x 0.35 0.45] [--z 0.25 0.35]
Writes data/results/center_start.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from arm.ik_v2 import DiffIK  # noqa: E402
from eval_reach_from_home import reach  # noqa: E402


def solve(q_home: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Joints for the flange at p with the home orientation, starting from the home pose."""
    ik = DiffIK(q_rest=q_home, max_joint_speed=1.0)
    R = ik.fk(q_home)[1]
    q = q_home.copy()
    for _ in range(1500):
        step = ik.step(q, p, R, 0.01)
        q = step.q_next
    return q, step.pos_err, step.rot_err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bases", type=float, default=0.98, help="m between the two arm bases")
    p.add_argument("--spacing", type=float, nargs="+", default=[0.3, 0.4], help="m between the two flanges")
    p.add_argument("--x", type=float, nargs="+", default=[0.35, 0.45])
    p.add_argument("--z", type=float, nargs="+", default=[0.25, 0.35])
    p.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.2, 0.25])
    args = p.parse_args()
    home = yaml.safe_load((REPO / "config" / "teleop_real.yaml").read_text())["home_q"]
    out = []
    for spacing, x, z in itertools.product(args.spacing, args.x, args.z):
        row = {"spacing": spacing, "x": x, "z": z, "arms": {}}
        for side, inward in (("left", -1.0), ("right", 1.0)):
            q_home = np.array(home[side], float)
            q, err, rot = solve(q_home, np.array([x, inward * (args.bases - spacing) / 2.0, z]))
            arm = {"q_deg": np.degrees(q).round(1).tolist(), "q_rad": q.round(4).tolist(),
                   "ik_err_mm": round(1000 * err, 1), "ik_rot_deg": round(float(np.degrees(rot)), 1),
                   "travel_from_home_deg": round(float(np.degrees(np.abs(q - q_home)).max()), 1)}
            if err < 0.002 and rot < 0.02:
                arm["reach"] = {str(r): reach(q, r, cap=1.0) for r in args.radii}
            row["arms"][side] = arm
        out.append(row)
        print(f"== flanges {spacing:.2f} m apart, x {x:.2f}, z {z:.2f}")
        for side, arm in row["arms"].items():
            print(f"  {side}: q deg {arm['q_deg']}, IK error {arm['ik_err_mm']} mm / {arm['ik_rot_deg']} deg, "
                  f"largest joint travel from home {arm['travel_from_home_deg']} deg")
            for r, res in arm.get("reach", {}).items():
                print(f"    r {r}: missed {res['missed_over_10mm']}/26, worst {res['worst_err_mm']} mm at {res['worst_direction']}, "
                      f"sigma min {res['sigma_min']}, joint offset max {max(res['joint_offset_max_rad'])}, "
                      f"limit margin {res['limit_margin_min_rad']}, z min {res['eef_z_min']}")
    path = REPO / "data" / "results" / "center_start.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
