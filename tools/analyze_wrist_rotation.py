#!/usr/bin/env python3
"""
analyze_wrist_rotation.py — how steady is the wrist orientation in a tools/record_run_v3.py log?

Per side: the rotation between consecutive wrist poses (frames less than 0.05 s apart), the
number of steps over 20 degrees (a flip), the angular speed, how far the hand turned from its
first pose, and the depth fusion's sources.

Usage (with .venv-arm):
    tools/analyze_wrist_rotation.py data/runs/<tag>/record.csv
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    rows: dict[str, list] = {}
    for r in csv.reader(open(sys.argv[1])):
        if r[0].endswith("_wrist_q"):
            rows.setdefault(r[0], []).append([float(v) for v in r[1:]])
        elif r[0].endswith("_diag"):
            rows.setdefault(r[0], []).append(json.loads(",".join(r[2:])).get("source"))
    for side in ("left", "right"):
        if f"{side}_wrist_q" not in rows:
            print(f"== {side}: no wrist orientation (recorded with record_run_v3.py?)")
            continue
        a = np.array(rows[f"{side}_wrist_q"])
        t, R = a[:, 0], Rotation.from_quat(a[:, 1:])
        near = np.diff(t) < 0.05
        step = np.degrees((R[:-1].inv() * R[1:]).magnitude())[near]
        speed = step / np.diff(t)[near]
        turned = np.degrees((R[0].inv() * R).magnitude())
        src = rows.get(f"{side}_diag", [])
        print(f"== {side}: {len(t)} poses")
        print(f"  step between frames deg p50/p95/p99/max {np.percentile(step, [50, 95, 99, 100]).round(1).tolist()}, "
              f"steps over 20 deg: {int((step > 20).sum())}")
        print(f"  angular speed deg/s p50/p95 {np.percentile(speed, [50, 95]).round(0).tolist()}, "
              f"turned from the first pose deg p50/p95/max {np.percentile(turned, [50, 95, 100]).round(0).tolist()}")
        print("  sources: " + ", ".join(f"{k} {100 * src.count(k) / len(src):.0f}%" for k in sorted(set(src))) if src else "  sources: none")


if __name__ == "__main__":
    main()
