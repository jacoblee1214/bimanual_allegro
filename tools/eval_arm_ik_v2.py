#!/usr/bin/env python3
"""
eval_arm_ik_v2.py — M1's offline evaluation with the wrist gate (arm/gate.py) in front of IK.

v1 (tools/eval_arm_ik.py) fed every wrist sample to the follower. On the clip that runs
to the image edges the wrist jumps 5 cm / 0.5 rad in a frame and the joints are asked for
up to 18x their speed limit (docs/results/m1_arm_ik.md). Here each sample first goes
through WristGate:
  ACCEPT    follower target -> IK -> speed limiter, as in v1
  REJECT    the arm holds its last command
  REENGAGE  the follower re-latches this wrist pose to the arm's current pose (no jump)
and the same metrics are reported with and without the gate, side by side.

Tracking error is measured on accepted samples only: a rejected sample's target is the
outlier itself, and the point is not to follow it.

Usage:
    .venv-arm/bin/python tools/eval_arm_ik_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from arm.follow import RelativeFollower, limit_step  # noqa: E402
from arm.gate import Verdict, WristGate  # noqa: E402
from arm.ik import DEFAULT_URDF, ArmIK  # noqa: E402

TRAJECTORIES = ("v2_palm", "v2_palm2", "v2_back", "v2_cross")
SIDES = ("left", "right")


def evaluate(ik: ArmIK, t: np.ndarray, wrist_p: np.ndarray, wrist_q: np.ndarray, gate: WristGate | None) -> dict:
    wrist_R = Rotation.from_quat(wrist_q).as_matrix()
    q_cmd = ik.q_rest.copy()
    follower = RelativeFollower()
    counts = {v.value: 0 for v in Verdict}
    pos_mm, rot_rad, speeds, clipped = [], [], [], 0
    t_prev = t[0]

    for i in range(len(t)):
        dt = max(float(t[i] - t_prev), 1e-3) if i else 1 / 30
        t_prev = t[i]
        verdict = gate(float(t[i]), wrist_p[i], wrist_R[i]) if gate else (Verdict.REENGAGE if i == 0 else Verdict.ACCEPT)
        counts[verdict.value] += 1
        if verdict is Verdict.REJECT:
            continue
        if verdict is Verdict.REENGAGE:
            follower.engage(wrist_p[i], wrist_R[i], *ik.fk(q_cmd))

        p_target, R_target = follower.target(wrist_p[i], wrist_R[i])
        res = ik.solve(p_target, R_target, q_cmd)
        q_cand = res.q if res.converged else q_cmd
        speeds.append(np.abs(q_cand - q_cmd).max() / dt)
        q_next = limit_step(q_cmd, q_cand, ik.velocity_limit, dt)
        clipped += not np.allclose(q_next, q_cand, atol=1e-9)
        q_cmd = q_next

        p, R = ik.fk(q_cmd)
        pos_mm.append(np.linalg.norm(p_target - p) * 1000)
        rot_rad.append(np.linalg.norm(pin.log3(R_target @ R.T)))

    n = len(speeds)
    return {
        "samples": int(len(t)), **counts,
        "pos_err_mm_median": float(np.median(pos_mm)), "pos_err_mm_p95": float(np.percentile(pos_mm, 95)),
        "rot_err_rad_median": float(np.median(rot_rad)), "rot_err_rad_p95": float(np.percentile(rot_rad, 95)),
        "joint_speed_asked_max": float(np.max(speeds)), "joint_speed_asked_p95": float(np.percentile(speeds, 95)),
        "clipped_fraction": clipped / n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", nargs="+", default=list(TRAJECTORIES))
    p.add_argument("--out", type=Path, default=REPO / "data" / "results" / "m2_arm_ik_gate.json")
    args = p.parse_args()

    ik = ArmIK(DEFAULT_URDF)
    limit = float(ik.velocity_limit.max())
    print(f"joint speed limit {limit:.2f} rad/s; gate: 1.5 m/s, 10 rad/s, re-clutch after 0.3 s\n")
    print(f"{'traj':8s} {'side':5s} {'gate':4s} | {'acc/rej/reeng':>13s} | {'pos mm med/p95':>14s} | "
          f"{'rot rad med/p95':>15s} | {'speed asked max/p95':>19s} | clipped")
    results: dict = {}
    for name in args.trajectories:
        data = np.load(REPO / "data" / "trajectories" / f"{name}.npz")
        for side in SIDES:
            if f"{side}_t" not in data:
                continue
            for label, gate in (("off", None), ("on", WristGate())):
                m = evaluate(ik, data[f"{side}_t"], data[f"{side}_p"], data[f"{side}_q"], gate)
                results.setdefault(name, {}).setdefault(side, {})[label] = m
                print(f"{name:8s} {side:5s} {label:4s} | {m['accept']:4d}/{m['reject']:3d}/{m['reengage']:3d} | "
                      f"{m['pos_err_mm_median']:5.2f} / {m['pos_err_mm_p95']:6.2f} | "
                      f"{m['rot_err_rad_median']:6.4f} / {m['rot_err_rad_p95']:6.4f} | "
                      f"{m['joint_speed_asked_max']:8.2f} / {m['joint_speed_asked_p95']:6.2f} | {m['clipped_fraction']:5.1%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"joint_speed_limit_rad_s": limit, "trajectories": results}, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
