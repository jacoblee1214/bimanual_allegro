#!/usr/bin/env python3
"""
eval_depth_filter.py — milestone M4 offline evaluation of anisotropic wrist depth filters.

Replays recorded clips through the raw per-frame detector and side assignment
(no One Euro inside the tracker) and compares five position filter variants on
each clip and hand side:

  1. `raw`: unfiltered per-frame SQPnP wrist position (`vision/wrist_pose.py`).
  2. `one_euro`: current `_PoseFilter` position One Euro filter
     (`min_cutoff=1.0`, `beta=10.0`, reset after 0.3 s gap).
  3. `ray_kalman`: `RayKalman` with `sigma_lat=0.002` m, `sigma_ray=0.045` m,
     `accel=0.25` m/s^1.5 (tuned once on `v2_back` and held fixed across all clips).
  4. `hold_ray_kalman`: causal `HandSizeHold(window=45, back_only=False)` before
     PnP, followed by `RayKalman`.
  5. `hold_back_ray_kalman`: causal `HandSizeHold(window=45, back_only=True)`
     (scales world landmarks only when the palm normal from `palm_frame` points
     away from the camera) before PnP, followed by `RayKalman`.

Clips and intrinsics:
  * `data/clips/ego_labels.avi` with `config/ego_camera_info.yaml` and
    `vision/hand_tracker_ego.EgoHandTracker` (phases from `ego_labels.jsonl`).
  * `data/clips/v2_back.avi` and `data/clips/v2_palm2.avi` with
    `config/camera_nominal_640x480.yaml` and `vision/hand_tracker.HandTracker`.

Metrics per clip and side:
  * Still-stretch jitter (same definition as `tools/analyze_scale_jitter.py`:
    palm centroid step < 3 px/frame over 2 consecutive neighbours either side):
    median of `|d2|` split into camera-ray (depth) and lateral components, mm,
    plus 5-detection moving-median RMS deviation (`dev_ray_rms_mm`, `dev_lat_rms_mm`).
  * All-frame responsiveness / lag:
    - `lag_xcorr_ms`: fine-grid (1 ms) shift-matching lag of the lateral wrist
      component against `raw` over contiguous runs.
    - `first_move_lag_ms`: midpoint-crossing delay (ms) on the first >= 25 mm
      lateral move.
    - `step_p95_ray_mm`, `step_p95_lat_mm`: 95th percentile per-frame wrist step
      along the ray and laterally across consecutive detections.
  * Fraction of frames output (`output_fraction_active` over active phases where
    that hand is in the task, `output_fraction_all` over all clip frames, and
    `gate_accept_fraction` passing `arm/gate.WristGate` at 1.5 m/s).

Usage:
    ~/bimanual_allegro/.venv-vision/bin/python tools/eval_depth_filter.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from arm.gate import Verdict, WristGate  # noqa: E402
from vision import camera  # noqa: E402
from vision.depth_filter import (  # noqa: E402
    DEFAULT_ACCEL,
    DEFAULT_SIGMA_LAT,
    DEFAULT_SIGMA_RAY,
    DEFAULT_WINDOW,
    HandSizeHold,
    RayKalman,
    is_back_facing,
    palm_size,
)
from vision.filters import OneEuroFilter  # noqa: E402
from vision.hand_tracker import POSITION_FILTER, RESET_GAP_S, SIDES, TASKS_MODEL, HandTracker  # noqa: E402
from vision.hand_tracker_ego import EgoHandTracker  # noqa: E402
from vision.handedness import Detection, assign_sides  # noqa: E402
from vision.wrist_pose import solve_wrist_pose  # noqa: E402
from tools.analyze_scale_jitter import MEDIAN_HALF, STILL_HALF, jitter, split, still_mask, v5_style  # noqa: E402

OUT = REPO / "data" / "results" / "m4_depth_filter.json"

CLIP_SPECS = (
    ("ego_labels", REPO / "data" / "clips" / "ego_labels.avi", REPO / "config" / "ego_camera_info.yaml", True),
    ("v2_back", REPO / "data" / "clips" / "v2_back.avi", REPO / "config" / "camera_nominal_640x480.yaml", False),
    ("v2_palm2", REPO / "data" / "clips" / "v2_palm2.avi", REPO / "config" / "camera_nominal_640x480.yaml", False),
)

VARIANTS = (
    "raw",
    "one_euro",
    "ray_kalman",
    "hold_ray_kalman",
    "hold_back_ray_kalman",
)


def make_tracker(cal: camera.CameraCalibration, ego: bool, prefer_gpu: bool) -> tuple[HandTracker, bool]:
    """Create EgoHandTracker or HandTracker on GPU if available, falling back to CPU."""
    cls = EgoHandTracker if ego else HandTracker
    if prefer_gpu and TASKS_MODEL.exists():
        try:
            return cls(cal, gpu=True), True
        except RuntimeError:
            pass
    return cls(cal, gpu=False), False


def detect_clip(
    avi_path: Path, cal_path: Path, ego: bool, prefer_gpu: bool
) -> tuple[np.ndarray, list[str], list[dict], camera.CameraCalibration, bool]:
    cal = camera.load(cal_path)
    log_path = avi_path.with_suffix(".jsonl")
    log_rows = [json.loads(line) for line in log_path.open()] if log_path.exists() else []

    cap = cv2.VideoCapture(str(avi_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(log_rows) == n_video:
        ts = [float(r["t"]) for r in log_rows]
        phases = [str(r.get("phase", "both")) for r in log_rows]
    else:
        ts = [i / fps for i in range(n_video)]
        phases = ["both"] * n_video

    tracker, used_gpu = make_tracker(cal, ego=ego, prefer_gpu=prefer_gpu)
    rows: list[dict] = []
    for t in ts:
        ok, frame = cap.read()
        if not ok:
            break
        raw = tracker.detect(frame, t)
        dets = [Detection(r.label, r.score, r.image_landmarks[0, 0]) for r in raw]
        assigned = (
            tracker.sides(dets, t)
            if ego
            else assign_sides(dets, tracker.min_score, tracker.label_to_side)
        )
        rows.append({side: (raw[i].world_landmarks, raw[i].image_landmarks) for side, i in assigned.items()})
    tracker.close()
    cap.release()
    n = len(rows)
    return np.asarray(ts[:n], dtype=np.float64), phases[:n], rows, cal, used_gpu


def run_one_euro(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    f = OneEuroFilter(**POSITION_FILTER)
    out = np.empty_like(p)
    last_t = None
    for j in range(len(p)):
        if last_t is not None and t[j] - last_t > RESET_GAP_S:
            f.reset()
        out[j] = f(p[j], float(t[j]))
        last_t = float(t[j])
    return out


def run_ray_kalman(
    p: np.ndarray,
    t: np.ndarray,
    sigma_lat: float,
    sigma_ray: float,
    accel: float,
) -> np.ndarray:
    kf = RayKalman(sigma_lat=sigma_lat, sigma_ray=sigma_ray, accel=accel, reset_gap_s=RESET_GAP_S)
    return np.array([kf(p[j], float(t[j])) for j in range(len(p))])


def solve_positions(
    world: np.ndarray,
    image: np.ndarray,
    cal: camera.CameraCalibration,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.full((len(world), 3), np.nan)
    reproj = np.full(len(world), np.nan)
    for j in range(len(world)):
        pose = solve_wrist_pose(
            world[j], image[j], cal.image_size, cal.camera_matrix, side, cal.dist_coeffs
        )
        if pose is not None:
            p[j] = pose.position
            reproj[j] = pose.reprojection_px
    return p, reproj


def _lateral_trace(p: np.ndarray, r_med: float) -> np.ndarray:
    """Bearing vector scaled by median depth: purely lateral (invariant to radial scaling)."""
    return (p / np.linalg.norm(p, axis=1, keepdims=True)) * r_med


def xcorr_lateral_lag_ms(t: np.ndarray, p_raw: np.ndarray, p_filt: np.ndarray) -> float:
    """Fine-grid (1 ms) shift-matching lag of the lateral component against raw, in ms."""
    if np.allclose(p_raw, p_filt):
        return 0.0
    r_med = float(np.median(np.linalg.norm(p_raw, axis=1)))
    raw_lat = _lateral_trace(p_raw, r_med)
    filt_lat = _lateral_trace(p_filt, r_med)

    breaks = np.flatnonzero(np.diff(t) > RESET_GAP_S) + 1
    lags_s = np.arange(-80, 81) * 0.001
    sse = np.zeros(len(lags_s))
    valid = False
    for t_seg, r_seg, f_seg in zip(
        np.split(t, breaks), np.split(raw_lat, breaks), np.split(filt_lat, breaks)
    ):
        if len(t_seg) < 10 or (t_seg[-1] - t_seg[0]) < 0.30:
            continue
        tg = np.arange(t_seg[0] + 0.08, t_seg[-1] - 0.08, 0.001)
        if len(tg) < 10:
            continue
        valid = True
        r_g = np.column_stack([np.interp(tg, t_seg, r_seg[:, k]) for k in range(3)])
        for li, tau in enumerate(lags_s):
            f_g = np.column_stack([np.interp(tg + tau, t_seg, f_seg[:, k]) for k in range(3)])
            sse[li] += np.sum((f_g - r_g) ** 2)
    return float(lags_s[int(np.argmin(sse))] * 1000.0) if valid else 0.0


def first_big_move_lag_ms(t: np.ndarray, p_raw: np.ndarray, p_filt: np.ndarray) -> float:
    """Midpoint-crossing time lag (ms) on the first >= 25 mm lateral move."""
    if np.allclose(p_raw, p_filt):
        return 0.0
    r_med = float(np.median(np.linalg.norm(p_raw, axis=1)))
    raw_lat = _lateral_trace(p_raw, r_med)
    filt_lat = _lateral_trace(p_filt, r_med)

    breaks = np.flatnonzero(np.diff(t) > RESET_GAP_S) + 1
    for t_seg, r_seg, f_seg in zip(
        np.split(t, breaks), np.split(raw_lat, breaks), np.split(filt_lat, breaks)
    ):
        if len(t_seg) < 15:
            continue
        for i in range(len(t_seg) - 10):
            j = i + 10
            if t_seg[j] - t_seg[i] > 0.8:
                continue
            delta = r_seg[j] - r_seg[i]
            dist = float(np.linalg.norm(delta))
            if dist >= 0.025:
                d_unit = delta / dist
                end = min(len(t_seg), j + 6)
                r_proj = (r_seg[i:end] - r_seg[i]) @ d_unit
                f_proj = (f_seg[i:end] - r_seg[i]) @ d_unit
                t_win = t_seg[i:end]
                mid = 0.5 * dist
                tg = np.arange(t_win[0], t_win[-1], 0.0005)
                rg = np.interp(tg, t_win, r_proj)
                fg = np.interp(tg, t_win, f_proj)
                tr_idx = np.flatnonzero(rg >= mid)
                tf_idx = np.flatnonzero(fg >= mid)
                if len(tr_idx) and len(tf_idx):
                    return round(float((tg[tf_idx[0]] - tg[tr_idx[0]]) * 1000.0), 1)
    return 0.0


def evaluate_variant(
    p: np.ndarray,
    p_raw: np.ndarray,
    t: np.ndarray,
    frames: np.ndarray,
    still: np.ndarray,
    n_total_frames: int,
    n_active_frames: int,
    n_in_active: int,
) -> dict:
    jit = jitter(p, frames, still)
    cons = np.diff(frames) == 1
    dp = np.diff(p, axis=0)[cons]
    step_ray, step_lat = split(dp, p[:-1][cons])

    diff_from_raw = p - p_raw
    err_ray, err_lat = split(diff_from_raw, p_raw)

    gate = WristGate()
    I3 = np.eye(3)
    verdicts = [gate(float(t[k]), p[k], I3) for k in range(len(t))]
    n_accept = sum(v is Verdict.ACCEPT for v in verdicts)
    n_reengage = sum(v is Verdict.REENGAGE for v in verdicts)
    n_reject = sum(v is Verdict.REJECT for v in verdicts)
    n_out = len(p)

    return {
        "n_output": int(n_out),
        "output_fraction_all": round(float(n_out / n_total_frames), 4),
        "output_fraction_active": round(float(n_in_active / n_active_frames), 4),
        "gate_accept_fraction": round(float((n_accept + n_reengage) / n_out), 4),
        "gate_reject_count": int(n_reject),
        "gate_reengage_count": int(n_reengage),
        "d2_ray_still_mm": round(float(jit["d2_depth_mm"]), 3),
        "d2_lat_still_mm": round(float(jit["d2_lateral_mm"]), 3),
        "d2_total_still_mm": round(float(jit["d2_total_mm"]), 3),
        "dev_ray_rms_still_mm": round(float(jit["dev_depth_rms_mm"]), 3),
        "dev_lat_rms_still_mm": round(float(jit["dev_lateral_rms_mm"]), 3),
        "v5_all_frames_d2_mm": round(float(v5_style(p, frames)), 3),
        "step_p95_ray_mm": round(float(np.percentile(step_ray, 95) * 1000.0), 2),
        "step_p95_lat_mm": round(float(np.percentile(step_lat, 95) * 1000.0), 2),
        "step_p95_total_mm": round(float(np.percentile(np.linalg.norm(dp, axis=1), 95) * 1000.0), 2),
        "lag_xcorr_ms": round(float(xcorr_lateral_lag_ms(t, p_raw, p)), 1),
        "first_move_lag_ms": round(float(first_big_move_lag_ms(t, p_raw, p)), 1),
        "lat_rms_vs_raw_mm": round(float(np.sqrt(np.mean(err_lat ** 2)) * 1000.0), 3),
    }


def evaluate_side(
    ts: np.ndarray,
    phases: list[str],
    rows: list[dict],
    cal: camera.CameraCalibration,
    side: str,
    still_px: float,
    sigma_lat: float,
    sigma_ray: float,
    accel: float,
    window: int,
) -> dict:
    n_total = len(rows)
    active_set = {side, "both"}
    n_active_frames = sum(1 for ph in phases if ph in active_set)

    idx = np.array([i for i, r in enumerate(rows) if side in r], dtype=int)
    t = ts[idx]
    world = np.array([rows[i][side][0] for i in idx])
    image = np.array([rows[i][side][1] for i in idx])

    p_raw, reproj = solve_positions(world, image, cal, side)
    ok = np.isfinite(p_raw).all(axis=1)
    idx, t, world, image, p_raw, reproj = idx[ok], t[ok], world[ok], image[ok], p_raw[ok], reproj[ok]

    phase_counts = Counter(phases[i] for i in idx)
    n_in_active = sum(1 for i in idx if phases[i] in active_set)
    still = still_mask(image, idx, cal.image_size, still_px)

    sizes = np.array([palm_size(w) for w in world])
    back_flags = np.array([is_back_facing(w, side) for w in world])

    # 1. Raw
    # 2. One Euro
    p_euro = run_one_euro(p_raw, t)
    # 3. RayKalman
    p_rk = run_ray_kalman(p_raw, t, sigma_lat, sigma_ray, accel)
    # 4. HandSizeHold + RayKalman
    hold_all = HandSizeHold(window=window, back_only=False, side=side, reset_gap_s=RESET_GAP_S)
    world_hold = np.array([hold_all(world[j], side, float(t[j])) for j in range(len(t))])
    p_hold_raw, _ = solve_positions(world_hold, image, cal, side)
    p_hold_rk = run_ray_kalman(p_hold_raw, t, sigma_lat, sigma_ray, accel)
    # 5. HandSizeHold(back_only) + RayKalman
    hold_back = HandSizeHold(window=window, back_only=True, side=side, reset_gap_s=RESET_GAP_S)
    world_hold_back = np.array([hold_back(world[j], side, float(t[j])) for j in range(len(t))])
    p_hold_back_raw, _ = solve_positions(world_hold_back, image, cal, side)
    p_hold_back_rk = run_ray_kalman(p_hold_back_raw, t, sigma_lat, sigma_ray, accel)

    series = {
        "raw": p_raw,
        "one_euro": p_euro,
        "ray_kalman": p_rk,
        "hold_ray_kalman": p_hold_rk,
        "hold_back_ray_kalman": p_hold_back_rk,
    }

    variants_out = {
        vname: evaluate_variant(
            p, p_raw, t, idx, still, n_total, n_active_frames, n_in_active
        )
        for vname, p in series.items()
    }

    return {
        "n_clip_frames": int(n_total),
        "n_active_phase_frames": int(n_active_frames),
        "n_detections": int(len(idx)),
        "n_detections_in_active_phases": int(n_in_active),
        "detections_by_phase": dict(phase_counts),
        "n_still": int(still.sum()),
        "back_facing_fraction": round(float(np.mean(back_flags)), 4),
        "palm_size_median_mm": round(float(np.median(sizes) * 1000.0), 2),
        "palm_size_cv": round(float(np.std(sizes) / np.mean(sizes)), 4),
        "depth_median_m": round(float(np.median(np.linalg.norm(p_raw, axis=1))), 4),
        "reproj_median_px": round(float(np.median(reproj)), 3),
        "variants": variants_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cpu", action="store_true", help="Force legacy CPU detector instead of Tasks GPU")
    parser.add_argument("--still-px", type=float, default=3.0, help="Still threshold in px/frame (default 3.0)")
    parser.add_argument("--sigma-lat", type=float, default=DEFAULT_SIGMA_LAT)
    parser.add_argument("--sigma-ray", type=float, default=DEFAULT_SIGMA_RAY)
    parser.add_argument("--accel", type=float, default=DEFAULT_ACCEL)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    clips_out: dict[str, dict] = {}
    used_gpu_any = False

    header = (
        f"{'clip':10s} {'side':5s} {'variant':20s} | {'out (act/all)':>13s} | "
        f"{'still d2 ray/lat mm':>19s} | {'v5 d2':>6s} | {'p95 step ray/lat':>16s} | "
        f"{'lag xcorr/move':>14s} | {'gate rej':>8s}"
    )
    print(header)
    print("-" * len(header))

    for clip_name, avi_path, cal_path, is_ego in CLIP_SPECS:
        ts, phases, rows, cal, used_gpu = detect_clip(
            avi_path, cal_path, ego=is_ego, prefer_gpu=not args.cpu
        )
        used_gpu_any = used_gpu_any or used_gpu
        phase_totals = dict(Counter(phases))
        clips_out[clip_name] = {
            "clip_path": str(avi_path.relative_to(REPO)),
            "calibration_path": str(cal_path.relative_to(REPO)),
            "tracker": "EgoHandTracker" if is_ego else "HandTracker",
            "detector": "Tasks API HandLandmarker (GPU)" if used_gpu else "legacy solutions.hands (CPU)",
            "fx_px": round(float(cal.camera_matrix[0, 0]), 2),
            "fy_px": round(float(cal.camera_matrix[1, 1]), 2),
            "n_frames": int(len(rows)),
            "phase_frame_counts": phase_totals,
            "sides": {},
        }

        for side in SIDES:
            res = evaluate_side(
                ts,
                phases,
                rows,
                cal,
                side,
                still_px=args.still_px,
                sigma_lat=args.sigma_lat,
                sigma_ray=args.sigma_ray,
                accel=args.accel,
                window=args.window,
            )
            clips_out[clip_name]["sides"][side] = res
            for vname in VARIANTS:
                m = res["variants"][vname]
                out_str = f"{m['output_fraction_active']:5.1%}/{m['output_fraction_all']:5.1%}"
                d2_str = f"{m['d2_ray_still_mm']:6.2f} / {m['d2_lat_still_mm']:5.2f}"
                step_str = f"{m['step_p95_ray_mm']:6.1f} / {m['step_p95_lat_mm']:5.1f}"
                lag_str = f"{m['lag_xcorr_ms']:4.0f} / {m['first_move_lag_ms']:5.1f}ms"
                rej_str = f"{m['gate_reject_count']:3d}/{m['n_output']:3d}"
                print(
                    f"{clip_name:10s} {side:5s} {vname:20s} | {out_str:>13s} | "
                    f"{d2_str:>19s} | {m['v5_all_frames_d2_mm']:6.2f} | "
                    f"{step_str:>16s} | {lag_str:>14s} | {rej_str:>8s}"
                )
            print("-" * len(header))

    payload = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "tool": "tools/eval_depth_filter.py",
            "detector": "Tasks API HandLandmarker (GPU)" if used_gpu_any else "legacy solutions.hands (CPU)",
            "still_definition": (
                f"palm centroid (0,5,9,13,17) step < {args.still_px} px/frame for the detection and "
                f"{STILL_HALF} neighbours either side, all consecutive frames"
            ),
            "moving_median_window": 2 * MEDIAN_HALF + 1,
            "reset_gap_s": RESET_GAP_S,
            "one_euro_position": POSITION_FILTER,
            "ray_kalman_params": {
                "sigma_lat_m": args.sigma_lat,
                "sigma_ray_m": args.sigma_ray,
                "accel_m_s1_5": args.accel,
                "tuned_on": "v2_back (held fixed across ego_labels, v2_back, v2_palm2)",
            },
            "hand_size_hold_params": {
                "window_frames": args.window,
                "landmarks": [0, 5, 9, 13, 17],
                "back_only_rule": "vision.wrist_pose.palm_frame(world, side)[2, 0] > 0.0",
            },
        },
        "clips": clips_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
