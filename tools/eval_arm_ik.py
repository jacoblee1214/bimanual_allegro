#!/usr/bin/env python3
"""
eval_arm_ik.py — milestone M1 offline evaluation of xArm7 IK and relative following.

Replays each recorded wrist trajectory (data/trajectories/v2_*.npz) per hand side
from `q_rest`, engaging the relative follower on the first sample, solving 6D IK
warm-started from the previous commanded `q`, and clamping per-step joint speed
to the xArm7 URDF velocity limit (3.14 rad/s ≈ 180 deg/s on all 7 joints).

Reports the M1 metrics table (docs/03_arm_follow_design.md, section 4), a reach
rate sweep over scales {1.0, 1.5, 2.0}, and writes data/results/m1_arm_ik.json.

Usage:
    .venv-arm/bin/python tools/eval_arm_ik.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from arm.follow import RelativeFollower, limit_step  # noqa: E402
from arm.ik import DEFAULT_URDF, ArmIK  # noqa: E402

DEFAULT_TRAJECTORIES = ("v2_palm", "v2_palm2", "v2_back", "v2_cross")
SIDES = ("left", "right")
SCALES = (1.0, 1.5, 2.0)
LIMIT_MARGIN_RAD = 0.1


def evaluate_trajectory(
    ik: ArmIK,
    t: np.ndarray,
    wrist_p: np.ndarray,
    wrist_q: np.ndarray,
    fps: float,
    scale: float = 1.0,
) -> dict:
    """Run follower + IK + velocity limiter over one side's wrist trajectory."""
    n = len(t)
    wrist_R = Rotation.from_quat(wrist_q).as_matrix()

    q_cmd = ik.q_rest.copy()
    eef_p0, eef_R0 = ik.fk(q_cmd)

    follower = RelativeFollower(scale=scale)
    follower.engage(wrist_p[0], wrist_R[0], eef_p0, eef_R0)

    max_vel = ik.velocity_limit  # 3.14 rad/s from xarm7.urdf (180 deg/s = pi rad/s)

    converged_count = 0
    clipped_count = 0
    near_limit_count = 0
    pos_errs_mm = np.empty(n, dtype=np.float64)
    rot_errs_rad = np.empty(n, dtype=np.float64)
    joint_vels = np.empty((n, ik.model.nq), dtype=np.float64)
    ik_times_ms = np.empty(n, dtype=np.float64)

    for i in range(n):
        dt = (1.0 / fps) if i == 0 else float(t[i] - t[i - 1])
        if dt <= 0.0:
            dt = 1.0 / fps

        p_target, R_target = follower.target(wrist_p[i], wrist_R[i])

        t0 = time.perf_counter()
        res = ik.solve(p_target, R_target, q_cmd)
        ik_times_ms[i] = (time.perf_counter() - t0) * 1000.0

        if res.converged:
            converged_count += 1
            q_cand = res.q
        else:
            q_cand = q_cmd

        joint_vels[i] = np.abs(q_cand - q_cmd) / dt
        q_next = limit_step(q_cmd, q_cand, max_vel, dt)
        if not np.allclose(q_next, q_cand, atol=1e-9):
            clipped_count += 1
        q_cmd = q_next

        margin = np.minimum(q_cmd - ik.q_min, ik.q_max - q_cmd)
        if float(margin.min()) < LIMIT_MARGIN_RAD:
            near_limit_count += 1

        p_cmd, R_cmd = ik.fk(q_cmd)
        pos_errs_mm[i] = float(np.linalg.norm(p_target - p_cmd)) * 1000.0
        rot_errs_rad[i] = float(np.linalg.norm(pin.log3(R_target @ R_cmd.T)))

    vel_max_per_joint = joint_vels.max(axis=0)
    vel_p95_per_joint = np.percentile(joint_vels, 95, axis=0)

    return {
        "samples": int(n),
        "converged": int(converged_count),
        "reach_rate": float(converged_count / n),
        "pos_err_mm_median": float(np.median(pos_errs_mm)),
        "pos_err_mm_p95": float(np.percentile(pos_errs_mm, 95)),
        "rot_err_rad_median": float(np.median(rot_errs_rad)),
        "rot_err_rad_p95": float(np.percentile(rot_errs_rad, 95)),
        "joint_vel_max_per_joint_rad_s": [round(float(x), 4) for x in vel_max_per_joint],
        "joint_vel_p95_per_joint_rad_s": [round(float(x), 4) for x in vel_p95_per_joint],
        "joint_vel_max_rad_s": float(vel_max_per_joint.max()),
        "joint_vel_p95_rad_s": float(vel_p95_per_joint.max()),
        "clipped_samples": int(clipped_count),
        "clipped_fraction": float(clipped_count / n),
        "near_limit_samples": int(near_limit_count),
        "near_limit_fraction": float(near_limit_count / n),
        "ik_ms_median": float(np.median(ik_times_ms)),
        "ik_ms_p95": float(np.percentile(ik_times_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectories",
        nargs="+",
        default=list(DEFAULT_TRAJECTORIES),
        help="trajectory stems under data/trajectories (default: v2_palm v2_palm2 v2_back v2_cross)",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="path to xArm7 URDF")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "results" / "m1_arm_ik.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    ik = ArmIK(args.urdf)
    p_rest, R_rest = ik.fk(ik.q_rest)

    print(f"URDF        : {args.urdf}")
    print(f"q_rest (rad): {np.round(ik.q_rest, 4).tolist()}")
    print(f"FK(q_rest) p: {np.round(p_rest, 4).tolist()} m")
    print(f"FK(q_rest) R: {np.round(R_rest, 4).tolist()}")
    print(f"vel limit   : {ik.velocity_limit[0]:.2f} rad/s (from URDF <limit velocity=\"3.14\"/>, ~180 deg/s)")
    print()

    header = (
        f"{'traj':8s} {'side':5s} | {'reach':>11s} | {'pos mm med/p95':>14s} | "
        f"{'rot rad med/p95':>15s} | {'vel max/p95':>12s} | {'clipped':>11s} | "
        f"{'near lim':>8s} | {'IK ms med/p95':>13s}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}
    scale_sweep: dict[str, dict[str, dict[str, float]]] = {}

    for name in args.trajectories:
        npz_path = REPO / "data" / "trajectories" / f"{name}.npz"
        data = np.load(npz_path)
        fps = float(data["fps"])
        results[name] = {}
        scale_sweep[name] = {}

        for side in SIDES:
            if f"{side}_t" not in data:
                continue
            t = data[f"{side}_t"]
            wp = data[f"{side}_p"]
            wq = data[f"{side}_q"]

            m = evaluate_trajectory(ik, t, wp, wq, fps=fps, scale=1.0)
            results[name][side] = m

            reach_str = f"{m['converged']:3d}/{m['samples']:3d} {m['reach_rate']:3.0%}"
            pos_str = f"{m['pos_err_mm_median']:5.2f} / {m['pos_err_mm_p95']:6.2f}"
            rot_str = f"{m['rot_err_rad_median']:6.4f} / {m['rot_err_rad_p95']:6.4f}"
            vel_str = f"{m['joint_vel_max_rad_s']:5.2f} / {m['joint_vel_p95_rad_s']:4.2f}"
            clip_str = f"{m['clipped_samples']:3d}/{m['samples']:3d} {m['clipped_fraction']:3.0%}"
            lim_str = f"{m['near_limit_fraction']:7.1%}"
            ik_str = f"{m['ik_ms_median']:5.3f} / {m['ik_ms_p95']:5.3f}"
            print(
                f"{name:8s} {side:5s} | {reach_str:>11s} | {pos_str:>14s} | "
                f"{rot_str:>15s} | {vel_str:>12s} | {clip_str:>11s} | "
                f"{lim_str:>8s} | {ik_str:>13s}"
            )

            scale_sweep[name][side] = {}
            for s in SCALES:
                ms = m if s == 1.0 else evaluate_trajectory(ik, t, wp, wq, fps=fps, scale=s)
                scale_sweep[name][side][f"{s:.1f}"] = round(ms["reach_rate"], 4)

    print("\nPer-joint speed before limiting (max / p95 in rad/s, j1..j7):")
    for name, sides_dict in results.items():
        for side, m in sides_dict.items():
            pairs = [
                f"j{j + 1}:{mx:.2f}/{p95:.2f}"
                for j, (mx, p95) in enumerate(
                    zip(m["joint_vel_max_per_joint_rad_s"], m["joint_vel_p95_per_joint_rad_s"])
                )
            ]
            print(f"  {name:8s} {side:5s}: " + "  ".join(pairs))

    print("\nScale sweep (reach rate):")
    print(f"{'traj':8s} {'side':5s} | " + " | ".join(f"scale={s:.1f}" for s in SCALES))
    print("-" * 46)
    for name, sides_dict in scale_sweep.items():
        for side, sdict in sides_dict.items():
            cols = " | ".join(f"{sdict[f'{s:.1f}']:9.1%}" for s in SCALES)
            print(f"{name:8s} {side:5s} | {cols}")

    payload = {
        "urdf": str(args.urdf),
        "q_rest": [round(float(x), 6) for x in ik.q_rest],
        "fk_q_rest": {
            "p_m": [round(float(x), 6) for x in p_rest],
            "R": [[round(float( x), 6) for x in row] for row in R_rest],
        },
        "velocity_limit_rad_s": [round(float(x), 4) for x in ik.velocity_limit],
        "limit_margin_rad": LIMIT_MARGIN_RAD,
        "trajectories": results,
        "scale_sweep_reach_rate": scale_sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
