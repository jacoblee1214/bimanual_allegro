#!/usr/bin/env python3
"""
eval_depth_wrist.py — milestone M5: the PnP wrist corrected with the sensor's palm depth, offline.

Reads a tools/record_ego_depth.py clip and its detections (tools/extract_ego_detections.py,
data/results/m5_detections_<clip>.npz; MediaPipe is not run again) and compares, per
hand:

  raw        PnP wrist as the tracker solved it
  raw_rk     raw through RayKalman with the M4 defaults (sigma_ray 45 mm)
  pnp_scaled raw times the clip's median sensor-corrected scale: one fixed calibration
             per hand, chosen after the fact. Not causal; it shows how much of the gain
             a single per-operator scale would give, and puts PnP's jitter in the same
             units as the fused variants (PnP that places the hand too close also
             shrinks every motion and every wobble by the same factor)
  fused      vision/depth_wrist.DepthFusion: PnP scaled to the sensor's palm depth,
             gated, held scale when the gate says no
  fused_rk   fused through RayKalman whose sigma_ray is --sigma-ray-fused on frames
             the sensor corrected and the M4 45 mm on held / PnP frames

The sensor's palm depth is the reference for the PnP depth. Its own noise is measured
here, not assumed: the spread of the palm depth over the 'still' phase (about its mean,
and from frame-to-frame differences, which drift does not inflate).

Metrics per clip and side, written to data/results/m5_depth_wrist.json:
  coverage     share of found frames whose output the sensor corrected (and why not)
  scale        sensor / PnP palm depth: median, p5, p95, by phase and palm / back facing;
               the wrist offset this means along the ray, mm
  noise floor  sensor palm depth per phase: std, frame-to-frame std / sqrt 2, mm
               (meaningful only where the hand was really still)
  still        per variant, still phase: std about the mean and frame-to-frame p95,
               along the camera ray and across it, mm; and the same numbers per phase over
               only the frames the sensor corrected (the same frames for every variant)
  lag          per variant, shift (1 ms grid) that best matches the lateral trace to raw's,
               and the ray trace to its own unfiltered input, ms
  robot axes   per variant, 'free' phase: frame-to-frame p95 per robot axis (R_CB @ p), mm

Usage (with .venv-vision, from the repository root):
    .venv-vision/bin/python -B tools/eval_depth_wrist.py smoke ego_depth
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision import camera  # noqa: E402
from vision.depth_filter import (  # noqa: E402
    DEFAULT_ACCEL,
    DEFAULT_SIGMA_LAT,
    DEFAULT_SIGMA_RAY,
    RESET_GAP_S,
    RayKalman,
    is_back_facing,
)
from vision.depth_wrist import (  # noqa: E402
    HOLD_WINDOW,
    MIN_FILL,
    MIN_HOLD,
    MIN_PIXELS,
    PALM,
    REL_BAND,
    SCALE_BAND,
    SHRINK,
    DepthFusion,
    hand_in_camera,
    palm_depth,
)

OUT = REPO / "data" / "results" / "m5_depth_wrist.json"
SIDES = ("left", "right")
VARIANTS = ("raw", "raw_rk", "pnp_scaled", "fused", "fused_rk")
#: Fused-RayKalman ray noise, metres, set once and kept for every clip: the std of the
#: sensor palm depth over smoke's 'ready' phase, right hand (a seated person's still hand,
#: 4.5 mm), rounded up. It was meant to come from ego_depth's still phase; that clip
#: was not there in time (docs/results/m5_depth_wrist.md).
SIGMA_RAY_FUSED = 0.005
#: Consecutive frames: a gap longer than this splits a run (the recorder runs at ~30 fps).
MAX_STEP_S = 0.05


def r(x: float, n: int = 2) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def pct(x: np.ndarray, qs=(5, 50, 95), n: int = 3) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"p{q}": None for q in qs} | {"n": 0}
    return {f"p{q}": r(np.percentile(x, q), n) for q in qs} | {"n": int(len(x))}


def split(v: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Component of v along the camera ray to p (signed) and the size of the rest, row-wise."""
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    along = np.sum(v * u, axis=1)
    return along, np.linalg.norm(v - along[:, None] * u, axis=1)


def steps(t: np.ndarray) -> np.ndarray:
    """Mask over np.diff rows: True where the two frames are consecutive."""
    return np.diff(t) < MAX_STEP_S


def run_kalman(p: np.ndarray, t: np.ndarray, sigma_ray: np.ndarray) -> np.ndarray:
    kf = RayKalman(sigma_lat=DEFAULT_SIGMA_LAT, sigma_ray=DEFAULT_SIGMA_RAY, accel=DEFAULT_ACCEL)
    out = np.empty_like(p)
    for j in range(len(p)):
        kf.sigma_ray = float(sigma_ray[j])  # read on every call; per frame, as a live tracker would
        out[j] = kf(p[j], float(t[j]))
    return out


def lag_ms(t: np.ndarray, ref: np.ndarray, x: np.ndarray) -> float | None:
    """Shift (ms, 1 ms grid, +-150 ms) that best matches x to ref over runs of >= 0.3 s."""
    ref, x = ref.reshape(len(t), -1), x.reshape(len(t), -1)
    if np.allclose(ref, x):
        return 0.0
    breaks = np.flatnonzero(np.diff(t) > RESET_GAP_S) + 1
    shifts = np.arange(-150, 151) * 0.001
    sse = np.zeros(len(shifts))
    used = False
    for ts, rs, xs in zip(np.split(t, breaks), np.split(ref, breaks), np.split(x, breaks)):
        if len(ts) < 10 or ts[-1] - ts[0] < 0.3:
            continue
        grid = np.arange(ts[0] + 0.15, ts[-1] - 0.15, 0.001)
        if len(grid) < 10:
            continue
        used = True
        rg = np.column_stack([np.interp(grid, ts, rs[:, k]) for k in range(rs.shape[1])])
        for i, tau in enumerate(shifts):
            xg = np.column_stack([np.interp(grid + tau, ts, xs[:, k]) for k in range(xs.shape[1])])
            sse[i] += np.sum((xg - rg) ** 2)
    return round(float(shifts[int(np.argmin(sse))] * 1000.0), 1) if used else None


def still_jitter(p: np.ndarray, t: np.ndarray, still: np.ndarray) -> dict:
    """mm: std about the still-phase mean and frame-to-frame p95, along the ray and across it."""
    if still.sum() < 10:
        return {"n": int(still.sum())}
    ps, ts = p[still], t[still]
    dev_ray, dev_lat = split(ps - ps.mean(axis=0), ps)
    cons = steps(ts)
    d_ray, d_lat = split(np.diff(ps, axis=0)[cons], ps[:-1][cons])
    return {
        "n": int(still.sum()),
        "std_ray_mm": r(np.std(dev_ray) * 1000),
        "std_lat_mm": r(np.sqrt(np.mean(dev_lat ** 2)) * 1000),
        "step_p95_ray_mm": r(np.percentile(np.abs(d_ray), 95) * 1000),
        "step_p95_lat_mm": r(np.percentile(d_lat, 95) * 1000),
    }


def robot_steps(p: np.ndarray, t: np.ndarray, mask: np.ndarray, r_cb: np.ndarray) -> dict:
    """'free'-phase frame-to-frame p95 per robot axis (x forward, y left, z up), mm."""
    pm, tm = p[mask], t[mask]
    cons = steps(tm)
    if cons.sum() < 10:
        return {"n_steps": int(cons.sum())}
    d = np.abs(np.diff(pm @ r_cb.T, axis=0)[cons]) * 1000
    return {"n_steps": int(cons.sum())} | {f"step_p95_{a}_mm": r(np.percentile(d[:, k], 95)) for k, a in enumerate("xyz")}


def reconstruction_error(world, image, p_raw, rot, k_pnp, size, side: str) -> float:
    """Largest gap (m) between hand_in_camera and the hand SQPnP places, solved again, every 10th frame."""
    worst = 0.0
    for j in range(0, len(world), 10):
        _, rvec, tvec = cv2.solvePnP(world[j], image[j, :, :2] * size, k_pnp, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
        placed = world[j] @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
        worst = max(worst, float(np.abs(placed - hand_in_camera(p_raw[j], rot[j], world[j], side)).max()))
    return worst


def noise_floor(sensor_z: np.ndarray, pnp_z: np.ndarray, t: np.ndarray, mask: np.ndarray) -> dict:
    """Spread of the sensor palm depth (and of PnP's, for scale) over the frames in mask, mm."""
    sz, pz, st = sensor_z[mask], pnp_z[mask], t[mask]
    cons = steps(st)
    out = {"n": int(len(sz))}
    if len(sz) < 10 or cons.sum() < 5:
        return out
    return out | {
        "sensor_z_mean_m": r(sz.mean(), 4),
        "sensor_z_std_mm": r(np.std(sz) * 1000),
        "sensor_z_step_std_over_sqrt2_mm": r(np.std(np.diff(sz)[cons]) / np.sqrt(2) * 1000),
        "pnp_palm_z_std_mm": r(np.std(pz) * 1000),
        "pnp_palm_z_step_std_over_sqrt2_mm": r(np.std(np.diff(pz)[cons]) / np.sqrt(2) * 1000),
    }


def evaluate_side(det, depth, depth_scale, k, k_pnp, r_cb, side: str, sigma_ray_fused: float) -> dict:
    found = det[f"{side}_found"] & np.isfinite(det[f"{side}_p"]).all(axis=1)
    idx = np.flatnonzero(found)
    out: dict = {"n_found": int(len(idx))}
    if len(idx) < 10:
        return out
    t, phase = det["t"][idx], det["phase"][idx]
    p_raw = det[f"{side}_p"][idx]
    rot, world, image = det[f"{side}_R"][idx], det[f"{side}_world"][idx], det[f"{side}_image"][idx]

    fusion = DepthFusion()
    n = len(idx)
    sensor_z, sensor_n, sensor_fill = np.full(n, np.nan), np.zeros(n, int), np.full(n, np.nan)
    pnp_palm_z, measured, used = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    p_fused, source, back = np.empty_like(p_raw), [], np.zeros(n, bool)
    for j, i in enumerate(idx):
        sensor = palm_depth(depth[i].astype(np.float64) * depth_scale, image[j])
        f = fusion(p_raw[j], rot[j], world[j], sensor, k, side)
        p_fused[j], used[j] = f.position, f.source == "depth"
        source.append(f.source)
        back[j] = is_back_facing(world[j], side)
        pnp_palm_z[j] = hand_in_camera(p_raw[j], rot[j], world[j], side)[list(PALM), 2].mean()
        if sensor is not None:
            sensor_z[j], sensor_n[j], sensor_fill[j] = sensor.z, sensor.n, sensor.fill
        if f.measured is not None:
            measured[j] = f.measured
    source = np.array(source)

    has_sensor = np.isfinite(sensor_z)
    in_band = (measured >= SCALE_BAND[0]) & (measured <= SCALE_BAND[1])
    out["coverage"] = {
        "fused_from_sensor": r(used.mean(), 4),
        "held_scale": r(np.mean(source == "hold"), 4),
        "raw_pnp": r(np.mean(source == "pnp"), 4),
        "no_sensor_value": r(np.mean(~has_sensor), 4),
        "edge_on": r(np.mean(has_sensor & ~np.isfinite(measured)), 4),
        "outside_abs_band": r(np.mean(np.isfinite(measured) & ~in_band), 4),
        "outside_rel_band": r(np.mean(in_band & ~used), 4),
        "by_phase_fused_from_sensor": {ph: r(used[phase == ph].mean(), 4) for ph in np.unique(phase)},
        "sensor_pixels_median": r(np.median(sensor_n[has_sensor]), 0) if has_sensor.any() else None,
        "sensor_fill_median": r(np.nanmedian(sensor_fill), 3) if has_sensor.any() else None,
    }
    m = np.isfinite(measured)
    wrist_dist = np.linalg.norm(p_raw, axis=1)
    out["scale"] = {
        "all": pct(measured),
        "used": pct(measured[used]),
        "by_phase": {ph: pct(measured[phase == ph]) for ph in np.unique(phase)},
        "back_facing": pct(measured[back]),
        "palm_facing": pct(measured[~back]),
        "back_facing_fraction": r(back.mean(), 4),
        "wrist_ray_offset_mm": pct((measured[m] - 1.0) * wrist_dist[m] * 1000, n=1),
        "sensor_palm_z_m": pct(sensor_z),
        "pnp_palm_z_m": pct(pnp_palm_z),
    }

    out["sensor_noise_floor"] = {ph: noise_floor(sensor_z, pnp_palm_z, t, (phase == ph) & has_sensor)
                                 for ph in np.unique(phase)}
    still = phase == "still"

    sig_raw = np.full(n, DEFAULT_SIGMA_RAY)
    sig_fused = np.where(used, sigma_ray_fused, DEFAULT_SIGMA_RAY)
    fixed = float(np.median(measured[used])) if used.any() else 1.0
    out["pnp_scaled_factor"] = r(fixed, 4)
    out["hand_in_camera_vs_sqpnp_max_m"] = reconstruction_error(world, image, p_raw, rot, k_pnp, depth.shape[2:0:-1], side)
    series = {"raw": p_raw, "raw_rk": run_kalman(p_raw, t, sig_raw), "pnp_scaled": p_raw * fixed,
              "fused": p_fused, "fused_rk": run_kalman(p_fused, t, sig_fused)}
    unfiltered = {"raw": p_raw, "raw_rk": p_raw, "pnp_scaled": p_raw * fixed, "fused": p_fused, "fused_rk": p_fused}
    r_med = float(np.median(wrist_dist))
    bearing = lambda p: p / np.linalg.norm(p, axis=1, keepdims=True) * r_med  # noqa: E731
    free = phase == "free"
    out["variants"] = {
        name: {
            "still": still_jitter(p, t, still),
            "sensor_frames_by_phase": {ph: still_jitter(p, t, (phase == ph) & used) for ph in np.unique(phase)},
            "lag_lateral_vs_raw_ms": lag_ms(t, bearing(p_raw), bearing(p)),
            "lag_ray_vs_unfiltered_ms": lag_ms(t, np.linalg.norm(unfiltered[name], axis=1), np.linalg.norm(p, axis=1)),
            "free_robot": robot_steps(p, t, free, r_cb),
        }
        for name, p in series.items()
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="clip names under data/clips/")
    ap.add_argument("--sigma-ray-fused", type=float, default=SIGMA_RAY_FUSED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    clips: dict = {}
    for name in args.clips:
        clip = REPO / "data" / "clips" / name
        det = np.load(REPO / "data" / "results" / f"m5_detections_{name}.npz")
        meta = json.loads((clip / "meta.json").read_text())
        depth = np.load(clip / "depth.npy", mmap_mode="r")
        if len(depth) != len(det["t"]):
            sys.exit(f"{name}: depth.npy has {len(depth)} rows, detections {len(det['t'])}")
        c = meta["color"]
        k = np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])
        r_cb = np.array(meta["R_CB"], dtype=np.float64)
        k_pnp = camera.load(REPO / "config" / "ego_camera_info.yaml").camera_matrix  # what the detections were solved with
        phases, counts = np.unique(det["phase"], return_counts=True)
        clips[name] = {
            "n_frames": int(len(det["t"])),
            "fps": r((len(det["t"]) - 1) / (det["t"][-1] - det["t"][0]), 2),
            "phase_frames": {str(ph): int(cn) for ph, cn in zip(phases, counts)},
            "depth_scale": meta["depth_scale"],
            "depth_preset": meta.get("depth_preset"),
            "sides": {s: evaluate_side(det, depth, meta["depth_scale"], k, k_pnp, r_cb, s, args.sigma_ray_fused) for s in SIDES},
        }
        for s in SIDES:
            res = clips[name]["sides"][s]
            if "variants" not in res:
                print(f"{name:10s} {s:5s} found {res['n_found']}: too few")
                continue
            cov, sc = res["coverage"], res["scale"]["all"]
            print(f"{name:10s} {s:5s} found {res['n_found']:4d}  fused {cov['fused_from_sensor']:.1%}  "
                  f"scale p5/50/95 {sc['p5']}/{sc['p50']}/{sc['p95']}  still floor {res['sensor_noise_floor'].get('still')}")
            for v in VARIANTS:
                m = res["variants"][v]
                print(f"    {v:9s} still {m['still']}  lag lat/ray {m['lag_lateral_vs_raw_ms']}/{m['lag_ray_vs_unfiltered_ms']}  "
                      f"free {m['free_robot']}")

    payload = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "tool": "tools/eval_depth_wrist.py",
            "detections": "data/results/m5_detections_<clip>.npz (tools/extract_ego_detections.py)",
            "intrinsics": "meta.json colour intrinsics (the stream depth is aligned to)",
            "palm_depth": {"landmarks": list(PALM), "shrink": SHRINK, "min_pixels": MIN_PIXELS, "min_fill": MIN_FILL,
                           "rejection": "valid 0.1-3.0 m, then median +- max(3 * 1.4826 MAD, 5 mm)"},
            "gate": {"abs_band": list(SCALE_BAND), "rel_band": REL_BAND, "min_hold": MIN_HOLD, "hold_window": HOLD_WINDOW},
            "ray_kalman": {"sigma_lat_m": DEFAULT_SIGMA_LAT, "sigma_ray_m": DEFAULT_SIGMA_RAY, "accel": DEFAULT_ACCEL,
                           "sigma_ray_fused_m": args.sigma_ray_fused,
                           "fused_rule": "sigma_ray_fused on sensor-corrected frames, sigma_ray on held / PnP frames"},
            "still": "frames whose recorder phase is 'still'",
            "consecutive": f"frame pairs less than {MAX_STEP_S} s apart",
            "robot_axes": "R_CB @ p, R_CB from the clip's meta.json",
        },
        "clips": clips,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
