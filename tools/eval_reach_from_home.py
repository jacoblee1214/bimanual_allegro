#!/usr/bin/env python3
"""
eval_reach_from_home.py — how far can each arm follow from the home pose, and how fast?

Offline, no robot. For the step from "x0.5 within 7 cm" to 1:1 following (docs/12_same_motion.md).
The same IK as arm/arm_follow_node_v6.py, position only, starting at config/teleop_real.yaml's
home_q, without a joint box.

  reach   the EEF target jumps to 26 directions x each radius and the IK runs for 8 s:
          the position error left, the smallest singular value on the way, how far every
          joint went from the start (the joint box that radius needs), the margin to the
          joint limits, and the EEF's lowest z and its y towards the other arm.
  speed   the target swings +-radius along x, y, z at --hz, for each joint-speed cap:
          the tracking error (RMS and worst) and the lag of the EEF behind the target.

Usage (with .venv-arm):
    tools/eval_reach_from_home.py [--radii 0.07 0.15 0.2 0.25 0.3] [--caps 0.3 0.6 1.0 1.5] [--hz 0.5]
Writes data/results/reach_from_home.json.
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
from arm.ik_v2 import DiffIK  # noqa: E402

DT = 0.01
DIRECTIONS = [np.array(d, float) / np.linalg.norm(d)
              for d in itertools.product((-1, 0, 1), repeat=3) if any(d)]


def reach(q0: np.ndarray, radius: float, cap: float) -> dict:
    ik = DiffIK(q_rest=q0, max_joint_speed=cap)
    p0, R0 = ik.fk(q0)
    rows = []
    for d in DIRECTIONS:
        q, sigma, offset, z_min, y_lo, y_hi = q0.copy(), np.inf, np.zeros(7), np.inf, np.inf, -np.inf
        for _ in range(int(8.0 / DT)):
            step = ik.step(q, p0 + radius * d, R0, DT)
            q, sigma = step.q_next, min(sigma, step.sigma_min)
            offset = np.maximum(offset, np.abs(q - q0))
            p = ik.fk(q)[0]
            z_min, y_lo, y_hi = min(z_min, p[2]), min(y_lo, p[1]), max(y_hi, p[1])
        margin = float(np.minimum(q - ik.q_min, ik.q_max - q).min())
        rows.append(dict(d=d.round(2).tolist(), err_mm=1000 * step.pos_err, sigma=sigma, offset=offset,
                         margin=margin, z_min=z_min, y_lo=y_lo, y_hi=y_hi))
    worst = max(rows, key=lambda r: r["err_mm"])
    return {
        "p_start": p0.round(3).tolist(),
        "missed_over_10mm": sum(r["err_mm"] > 10.0 for r in rows),
        "worst_err_mm": round(worst["err_mm"], 1), "worst_direction": worst["d"],
        "sigma_min": round(min(r["sigma"] for r in rows), 3),
        "joint_offset_max_rad": np.max([r["offset"] for r in rows], axis=0).round(2).tolist(),
        "limit_margin_min_rad": round(min(r["margin"] for r in rows), 2),
        "eef_z_min": round(min(r["z_min"] for r in rows), 3),
        "eef_y_range": [round(min(r["y_lo"] for r in rows), 3), round(max(r["y_hi"] for r in rows), 3)],
    }


def speed(q0: np.ndarray, radius: float, cap: float, hz: float) -> dict:
    out = {}
    for axis, name in enumerate("xyz"):
        ik = DiffIK(q_rest=q0, max_joint_speed=cap)
        p0, R0 = ik.fk(q0)
        t = np.arange(0.0, 3.0 / hz, DT)
        want = radius * np.sin(2 * np.pi * hz * t)
        q, got = q0.copy(), []
        for w in want:
            target = p0.copy()
            target[axis] += w
            q = ik.step(q, target, R0, DT).q_next
            got.append(ik.fk(q)[0][axis] - p0[axis])
        got = np.array(got)
        keep = t >= 1.0 / hz                                   # skip the first cycle
        lags = np.arange(0, int(1.0 / DT))
        errors = [np.sqrt(np.mean((got[keep] - np.roll(want, k)[keep]) ** 2)) for k in lags]
        out[name] = {"rms_mm": round(1000 * float(np.sqrt(np.mean((got - want)[keep] ** 2))), 1),
                     "worst_mm": round(1000 * float(np.abs(got - want)[keep].max()), 1),
                     "lag_ms": int(1000 * DT * lags[int(np.argmin(errors))])}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radii", type=float, nargs="+", default=[0.07, 0.15, 0.2, 0.25, 0.3])
    p.add_argument("--caps", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5])
    p.add_argument("--hz", type=float, default=0.5, help="swing frequency of the speed test (default 0.5)")
    args = p.parse_args()
    home = yaml.safe_load((REPO / "config" / "teleop_real.yaml").read_text())["home_q"]
    result = {"reach": {}, "speed": {}, "hz": args.hz}
    for side in ("left", "right"):
        q0 = np.array(home[side], float)
        print(f"== {side}: reach (26 directions, position only, no joint box)")
        for r in args.radii:
            result["reach"][f"{side}/{r}"] = row = reach(q0, r, cap=1.0)
            print(f"  r {r:.2f}: missed {row['missed_over_10mm']}/26, worst {row['worst_err_mm']} mm at {row['worst_direction']}, "
                  f"sigma min {row['sigma_min']}, joint offsets {row['joint_offset_max_rad']}, "
                  f"limit margin {row['limit_margin_min_rad']}, z min {row['eef_z_min']}, y {row['eef_y_range']}")
        print(f"== {side}: speed (swing +-r at {args.hz} Hz; peak hand speed = 2 pi hz r)")
        for r in args.radii:
            for cap in args.caps:
                result["speed"][f"{side}/{r}/{cap}"] = row = speed(q0, r, cap, args.hz)
                print(f"  r {r:.2f} cap {cap:.1f}: " + "  ".join(
                    f"{k} rms {v['rms_mm']} worst {v['worst_mm']} mm lag {v['lag_ms']} ms" for k, v in row.items()))
    out = REPO / "data" / "results" / "reach_from_home.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
