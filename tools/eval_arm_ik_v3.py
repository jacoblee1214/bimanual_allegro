#!/usr/bin/env python3
"""
eval_arm_ik_v3.py — milestone M2 evaluation comparing v1 iterative IK + clip against
v2 velocity-level differential IK (`arm/ik_v2.py`) behind `WristGate` (`arm/gate.py`).

Per trajectory and hand side, evaluates three configurations:
  A) v1 solver + clip (`ArmIK` + `limit_step` at 3.14 rad/s, with `WristGate`)
  B) `DiffIK` at `max_joint_speed = 3.14` rad/s (URDF velocity limit)
  C) `DiffIK` at `max_joint_speed = 1.0` rad/s (first real-robot setting candidate)

`DiffIK` parameters (`gain=1.0`, `rot_weight=0.8`, `null_gain=0.05`, `damping=0.10`,
`sigma_threshold=0.12`, `limit_margin=0.05`) are tuned once on `v2_palm2` and `v2_back`
(inside the camera operating area) and held fixed across all trajectories:
  - `gain = 1.0`: takes a full linearized step per frame when below the velocity cap,
    minimizing steady-state lag on smooth trajectories (`v2_palm2`, `v2_back`).
  - `rot_weight = 0.8`: slightly prioritizes position over orientation when velocity
    scaling is active while keeping orientation tracking below 0.015 rad median on
    in-bounds clips.
  - `null_gain = 0.05`: pulls the redundant 7th DOF toward `q_rest` without disturbing
    the primary EEF task or triggering velocity scaling at rest.
  - `damping = 0.10`, `sigma_threshold = 0.12`: since `sigma_min` on `v2_palm2` and
    `v2_back` stays in `0.151..0.192`, setting `sigma_threshold = 0.12` leaves normal
    operating poses completely undamped (`lambda = 0`, avoiding artificial tracking lag)
    while smoothly ramping damping up to `lambda_max = 0.10` when approaching wrist or
    shoulder singularities (`sigma_min < 0.12`, as in `v2_cross`).

Usage:
    .venv-arm/bin/python tools/eval_arm_ik_v3.py
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
from arm.ik import DEFAULT_Q_REST as Q_REST_V1, DEFAULT_URDF, ArmIK  # noqa: E402
from arm.ik_v2 import DEFAULT_Q_REST as Q_REST_V2, DiffIK  # noqa: E402

TRAJECTORIES = ("v2_palm", "v2_palm2", "v2_back", "v2_cross")
SIDES = ("left", "right")


def evaluate_v1(ik: ArmIK, t: np.ndarray, wrist_p: np.ndarray, wrist_q: np.ndarray) -> tuple[dict, list[dict]]:
    """Mode A: v1 iterative `ArmIK.solve` + per-joint `limit_step` with `WristGate`."""
    wrist_R = Rotation.from_quat(wrist_q).as_matrix()
    q_cmd = ik.q_rest.copy()
    follower = RelativeFollower()
    gate = WristGate()
    counts = {v.value: 0 for v in Verdict}
    pos_mm, rot_rad, speeds_sent, speeds_asked, sigmas, clipped = [], [], [], [], [], 0
    trace: list[dict] = []
    t_prev = t[0]

    for i in range(len(t)):
        dt = max(float(t[i] - t_prev), 1e-3) if i else 1.0 / 30.0
        t_prev = t[i]
        verdict = gate(float(t[i]), wrist_p[i], wrist_R[i])
        counts[verdict.value] += 1
        if verdict is Verdict.REJECT:
            continue
        if verdict is Verdict.REENGAGE:
            follower.engage(wrist_p[i], wrist_R[i], *ik.fk(q_cmd))

        p_target, R_target = follower.target(wrist_p[i], wrist_R[i])
        J = pin.computeFrameJacobian(
            ik.model, ik.data, q_cmd, ik.ee_id, pin.LOCAL_WORLD_ALIGNED
        )
        s_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        sigmas.append(s_min)

        res = ik.solve(p_target, R_target, q_cmd)
        q_cand = res.q if res.converged else q_cmd
        dq_asked = q_cand - q_cmd
        spd_asked = float(np.abs(dq_asked).max() / dt)
        speeds_asked.append(spd_asked)

        q_next = limit_step(q_cmd, q_cand, ik.velocity_limit, dt)
        dq_sent = q_next - q_cmd
        clipped += int(not np.allclose(q_next, q_cand, atol=1e-9))
        spd_sent = float(np.abs(dq_sent).max() / dt)
        speeds_sent.append(spd_sent)

        trace.append({
            "t": round(float(t[i]), 4),
            "q_before": q_cmd.copy(),
            "dq_asked": dq_asked.copy(),
            "dq_sent": dq_sent.copy(),
            "spd_asked": spd_asked,
            "spd_sent": spd_sent,
            "sigma_min": s_min,
        })
        q_cmd = q_next

        p, R = ik.fk(q_cmd)
        pos_mm.append(float(np.linalg.norm(p_target - p) * 1000.0))
        rot_rad.append(float(np.linalg.norm(pin.log3(R_target @ R.T))))

    n = len(speeds_sent)
    summary = {
        "samples": int(len(t)),
        **counts,
        "pos_err_mm_median": round(float(np.median(pos_mm)), 4),
        "pos_err_mm_p95": round(float(np.percentile(pos_mm, 95)), 4),
        "rot_err_rad_median": round(float(np.median(rot_rad)), 6),
        "rot_err_rad_p95": round(float(np.percentile(rot_rad, 95)), 6),
        "joint_speed_sent_max": round(float(np.max(speeds_sent)), 4),
        "joint_speed_sent_p95": round(float(np.percentile(speeds_sent, 95)), 4),
        "joint_speed_asked_max": round(float(np.max(speeds_asked)), 4),
        "joint_speed_asked_p95": round(float(np.percentile(speeds_asked, 95)), 4),
        "sigma_min_min": round(float(np.min(sigmas)), 4),
        "sigma_min_p5": round(float(np.percentile(sigmas, 5)), 4),
        "scaled_fraction": round(float(clipped / n), 4),
        "time_to_5mm_s": None,
    }
    return summary, trace


def evaluate_diff(ik: DiffIK, t: np.ndarray, wrist_p: np.ndarray, wrist_q: np.ndarray) -> tuple[dict, list[dict]]:
    """Modes B and C: single-step `DiffIK.step` with `WristGate`."""
    wrist_R = Rotation.from_quat(wrist_q).as_matrix()
    q_cmd = ik.q_rest.copy()
    follower = RelativeFollower()
    gate = WristGate()
    counts = {v.value: 0 for v in Verdict}
    pos_mm, rot_rad, speeds_sent, sigmas, scaled_count = [], [], [], [], 0
    accept_times, accept_pos_mm = [], []
    trace: list[dict] = []
    t_prev = t[0]

    for i in range(len(t)):
        dt = max(float(t[i] - t_prev), 1e-3) if i else 1.0 / 30.0
        t_prev = t[i]
        verdict = gate(float(t[i]), wrist_p[i], wrist_R[i])
        counts[verdict.value] += 1
        if verdict is Verdict.REJECT:
            continue
        if verdict is Verdict.REENGAGE:
            follower.engage(wrist_p[i], wrist_R[i], *ik.fk(q_cmd))

        p_target, R_target = follower.target(wrist_p[i], wrist_R[i])
        res = ik.step(q_cmd, p_target, R_target, dt)
        dq_sent = res.q_next - q_cmd
        spd_sent = float(np.abs(dq_sent).max() / dt)
        speeds_sent.append(spd_sent)
        sigmas.append(float(res.sigma_min))
        scaled_count += int(res.scaled)

        trace.append({
            "t": round(float(t[i]), 4),
            "q_before": q_cmd.copy(),
            "dq_sent": dq_sent.copy(),
            "spd_sent": spd_sent,
            "sigma_min": float(res.sigma_min),
        })
        q_cmd = res.q_next

        err_mm = float(res.pos_err * 1000.0)
        pos_mm.append(err_mm)
        rot_rad.append(float(res.rot_err))
        if verdict is Verdict.ACCEPT:
            accept_times.append(float(t[i] - t[0]))
            accept_pos_mm.append(err_mm)

    n = len(speeds_sent)
    time_to_5mm: float | str = next(
        (round(tm, 4) for tm, er in zip(accept_times, accept_pos_mm) if er <= 5.0),
        "never",
    )
    summary = {
        "samples": int(len(t)),
        **counts,
        "pos_err_mm_median": round(float(np.median(pos_mm)), 4),
        "pos_err_mm_p95": round(float(np.percentile(pos_mm, 95)), 4),
        "rot_err_rad_median": round(float(np.median(rot_rad)), 6),
        "rot_err_rad_p95": round(float(np.percentile(rot_rad, 95)), 6),
        "joint_speed_sent_max": round(float(np.max(speeds_sent)), 4),
        "joint_speed_sent_p95": round(float(np.percentile(speeds_sent, 95)), 4),
        "sigma_min_min": round(float(np.min(sigmas)), 4),
        "sigma_min_p5": round(float(np.percentile(sigmas, 5)), 4),
        "scaled_fraction": round(float(scaled_count / n), 4),
        "time_to_5mm_s": time_to_5mm,
    }
    return summary, trace


def summarize_singularity_window(trace: list[dict], has_asked: bool) -> dict:
    win = [r for r in trace if 4.30 <= r["t"] <= 4.93]
    q2_vals = [float(r["q_before"][1]) for r in win]
    q6_vals = [float(r["q_before"][5]) for r in win]
    j1_sent = [float(r["dq_sent"][0]) for r in win]
    j3_sent = [float(r["dq_sent"][2]) for r in win]
    spd_sent = [float(r["spd_sent"]) for r in win]
    sigmas = [float(r["sigma_min"]) for r in win]
    r_687 = next((r for r in trace if abs(r["t"] - 6.8667) < 0.01), None)

    out = {
        "t_4_3_to_4_9_q2_min_rad": round(min(q2_vals), 4),
        "t_4_3_to_4_9_q2_max_rad": round(max(q2_vals), 4),
        "t_4_3_to_4_9_q6_min_rad": round(min(q6_vals), 4),
        "t_4_3_to_4_9_q6_max_rad": round(max(q6_vals), 4),
        "t_4_3_to_4_9_sigma_min": round(min(sigmas), 4),
        "t_4_3_to_4_9_j1_dq_sent_max_rad": round(max(np.abs(j1_sent)), 4),
        "t_4_3_to_4_9_j3_dq_sent_max_rad": round(max(np.abs(j3_sent)), 4),
        "t_4_3_to_4_9_spd_sent_min_rad_s": round(min(spd_sent), 4),
        "t_4_3_to_4_9_spd_sent_max_rad_s": round(max(spd_sent), 4),
        "t_6_87_j6_dq_sent_rad": round(float(r_687["dq_sent"][5]), 4) if r_687 else None,
        "t_6_87_spd_sent_rad_s": round(float(r_687["spd_sent"]), 4) if r_687 else None,
    }
    if has_asked:
        j1_asked = [float(r["dq_asked"][0]) for r in win]
        j3_asked = [float(r["dq_asked"][2]) for r in win]
        spd_asked = [float(r["spd_asked"]) for r in win]
        out.update({
            "t_4_3_to_4_9_j1_dq_asked_max_rad": round(max(j1_asked), 4),
            "t_4_3_to_4_9_j3_dq_asked_min_rad": round(min(j3_asked), 4),
            "t_4_3_to_4_9_spd_asked_min_rad_s": round(min(spd_asked), 4),
            "t_4_3_to_4_9_spd_asked_max_rad_s": round(max(spd_asked), 4),
            "t_6_87_j6_dq_asked_rad": round(float(r_687["dq_asked"][5]), 4) if r_687 else None,
            "t_6_87_spd_asked_rad_s": round(float(r_687["spd_asked"]), 4) if r_687 else None,
        })
    return out


def posture_summary(ik: ArmIK | DiffIK, q: np.ndarray) -> dict:
    p, R = ik.fk(q)
    J = pin.computeFrameJacobian(ik.model, ik.data, q, ik.ee_id, pin.LOCAL_WORLD_ALIGNED)
    s = np.linalg.svd(J, compute_uv=False)
    margin = np.minimum(q - ik.q_min, ik.q_max - q)
    return {
        "q_rad": [round(float(x), 4) for x in q],
        "eef_pos_m": [round(float(x), 4) for x in p],
        "eef_rot": [[round(float(x), 4) for x in row] for row in R],
        "min_limit_margin_rad": round(float(margin.min()), 4),
        "singular_values": [round(float(x), 4) for x in s],
        "sigma_min": round(float(s[-1]), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectories", nargs="+", default=list(TRAJECTORIES))
    parser.add_argument(
        "--out", type=Path, default=REPO / "data" / "results" / "m2_diff_ik.json"
    )
    args = parser.parse_args()

    ik_a = ArmIK(DEFAULT_URDF)
    ik_b = DiffIK(DEFAULT_URDF)  # default max_joint_speed = URDF limit (3.14 rad/s)
    ik_c = DiffIK(DEFAULT_URDF, max_joint_speed=1.0)

    v1_info = posture_summary(ik_a, Q_REST_V1)
    v2_info = posture_summary(ik_b, Q_REST_V2)

    print(f"v1 DEFAULT_Q_REST: {v1_info['q_rad']} -> p={v1_info['eef_pos_m']} m, "
          f"sigma_min={v1_info['sigma_min']:.4f} (s={v1_info['singular_values']})")
    print(f"v2 DEFAULT_Q_REST: {v2_info['q_rad']} -> p={v2_info['eef_pos_m']} m, "
          f"sigma_min={v2_info['sigma_min']:.4f} (s={v2_info['singular_values']})")
    print(f"DiffIK params: gain={ik_b.gain}, rot_weight={ik_b.rot_weight}, "
          f"null_gain={ik_b.null_gain}, damping={ik_b.damping}, "
          f"sigma_threshold={ik_b.sigma_threshold}, limit_margin={ik_b.limit_margin}\n")

    header = (
        f"{'traj':8s} {'side':5s} {'mode':10s} | {'acc/rej/ree':>11s} | "
        f"{'pos mm med/p95':>14s} | {'rot rad med/p95':>15s} | "
        f"{'spd sent max/p95':>16s} | {'sigma min/p5':>12s} | {'scaled':>6s} | {'t<=5mm':>7s}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}
    singularity_window: dict[str, dict] = {}

    for name in args.trajectories:
        data = np.load(REPO / "data" / "trajectories" / f"{name}.npz")
        results[name] = {}
        for side in SIDES:
            if f"{side}_t" not in data:
                continue
            t = data[f"{side}_t"]
            wp = data[f"{side}_p"]
            wq = data[f"{side}_q"]

            m_a, tr_a = evaluate_v1(ik_a, t, wp, wq)
            m_b, tr_b = evaluate_diff(ik_b, t, wp, wq)
            m_c, tr_c = evaluate_diff(ik_c, t, wp, wq)

            modes = {
                "A_v1_clip": m_a,
                "B_diff_3.14": m_b,
                "C_diff_1.0": m_c,
            }
            results[name][side] = modes

            if name == "v2_cross" and side == "right":
                singularity_window = {
                    "A_v1_clip": summarize_singularity_window(tr_a, has_asked=True),
                    "B_diff_3.14": summarize_singularity_window(tr_b, has_asked=False),
                    "C_diff_1.0": summarize_singularity_window(tr_c, has_asked=False),
                }

            for mode_label, m in modes.items():
                t5_str = (
                    "-"
                    if m["time_to_5mm_s"] is None
                    else (
                        f"{m['time_to_5mm_s']:.2f}s"
                        if isinstance(m["time_to_5mm_s"], (int, float))
                        else str(m["time_to_5mm_s"])
                    )
                )
                print(
                    f"{name:8s} {side:5s} {mode_label:10s} | "
                    f"{m['accept']:3d}/{m['reject']:3d}/{m['reengage']:2d} | "
                    f"{m['pos_err_mm_median']:5.2f} / {m['pos_err_mm_p95']:6.2f} | "
                    f"{m['rot_err_rad_median']:6.4f} / {m['rot_err_rad_p95']:6.4f} | "
                    f"{m['joint_speed_sent_max']:7.2f} / {m['joint_speed_sent_p95']:6.2f} | "
                    f"{m['sigma_min_min']:5.3f}/{m['sigma_min_p5']:5.3f} | "
                    f"{m['scaled_fraction']:5.1%} | {t5_str:>7s}"
                )

    payload = {
        "urdf": str(DEFAULT_URDF),
        "gate": {
            "max_speed_m_s": 1.5,
            "max_angular_speed_rad_s": 10.0,
            "reacquire_gap_s": 0.3,
        },
        "q_rest_comparison": {
            "v1_arm_ik": v1_info,
            "v2_diff_ik": v2_info,
        },
        "diff_ik_parameters": {
            "gain": ik_b.gain,
            "rot_weight": ik_b.rot_weight,
            "null_gain": ik_b.null_gain,
            "damping": ik_b.damping,
            "sigma_threshold": ik_b.sigma_threshold,
            "limit_margin": ik_b.limit_margin,
            "max_joint_speed_B_rad_s": float(ik_b.max_joint_speed.max()),
            "max_joint_speed_C_rad_s": float(ik_c.max_joint_speed.max()),
        },
        "v2_cross_right_singularity_window": singularity_window,
        "trajectories": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
