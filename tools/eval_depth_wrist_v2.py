#!/usr/bin/env python3
"""
eval_depth_wrist_v2.py — milestone M6: fixes for the left wrist's PnP tilt flip, offline.

M5 (tools/eval_depth_wrist.py) put the PnP palm at the sensor's palm depth. The wrist's
depth relative to the palm still came from PnP's rotation, and on the left hand of
ego_depth_424 PnP flipped between two tilt solutions in 'still'. This compares fixes on
the same clips and detections (data/results/m5_detections_<clip>.npz; MediaPipe is not
run again), with M5's metric code imported from tools/eval_depth_wrist.py:

  raw        PnP wrist as the tracker solved it                         (M5 baseline)
  fused      vision/depth_wrist.DepthFusion                              (M5 baseline)
  fused_rk   fused through RayKalman, sigma_ray 5 mm on sensor frames    (M5 baseline)
  v2         vision/depth_wrist_v2.WristDepthV2: landmark 0's pixel ray cut by a plane
             fitted to the sensor's palm pixels; palm frame from the landmarks cut by
             that plane (fix a). Gate: v1's absolute band only
  v2_relgate v2 with v1's relative band added back (ablation)
  plane_pnpray  as v2, but PnP's wrist bearing cut by the plane (fix a, other ray)
  local      landmark 0's pixel ray at the median sensor depth of a disc around the
             wrist pixel; PnP's rotation (fix a, no plane)
  pnp_cont   PnP's two local minima (SQPnP's, and LM from the tilt mirrored about the
             line of sight); among those within CONT_TOL_PX of the best reprojection, the
             one nearest the previous frame's choice; then DepthFusion (fix b)
  pnp_sensor the minimum whose palm normal is nearest the sensor plane's; DepthFusion (fix b)
  pnp_align  PnP's hand turned about its palm centre by the smallest rotation that puts
             its palm normal on the sensor plane's; DepthFusion (fix c)
  pnp_align_absonly  pnp_align with DepthFusion's relative band switched off (ablation)
  <name>_rk  that through RayKalman, sigma_ray 5 mm on its sensor frames, 45 mm otherwise

The v2-like fixes (v2, plane_pnpray, local) gate their ratio (sensor wrist distance / PnP
wrist distance) with v1's absolute band only; the PnP-based fixes keep DepthFusion's full
gate. The two ablations swap these. Non-depth frames fall back on PnP times the held
ratio, as in v1; their palm frame is PnP's. Per side: M5's still / lag / free-robot metrics, plus
the palm frame's frame-to-frame angle (p95, degrees; still and free), the number of
'still' steps over FLIP_DEG (a still hand does not turn 20 degrees in 33 ms), the angle
between each variant's palm normal and the sensor plane's, and v2's call time.

Usage (with .venv-vision, from the repository root):
    .venv-vision/bin/python -B tools/eval_depth_wrist_v2.py ego_depth_424 ego_depth ego_depth_work
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.eval_depth_wrist import (  # noqa: E402
    MAX_STEP_S,
    SIGMA_RAY_FUSED,
    lag_ms,
    pct,
    r,
    robot_steps,
    run_kalman,
    steps,
    still_jitter,
)
from vision import camera  # noqa: E402
from vision.depth_filter import DEFAULT_SIGMA_RAY, palm_frame  # noqa: E402
from vision.depth_wrist import (  # noqa: E402
    HOLD_WINDOW,
    MIN_HOLD,
    PALM,
    REL_BAND,
    SCALE_BAND,
    DepthFusion,
    hand_in_camera,
    palm_depth,
)
from vision.depth_wrist_v2 import MIN_COS, WristDepthV2, on_plane, pixel_rays, sensor_plane  # noqa: E402

OUT = REPO / "data" / "results" / "m6_wrist_flip.json"
SIDES = ("left", "right")
#: A frame-to-frame palm frame turn larger than this in 'still' counts as a flip, degrees.
FLIP_DEG = 20.0
#: pnp_cont: minima within this much RMS reprojection of the best are treated as equal, pixels.
CONT_TOL_PX = 2.0
#: Two minima closer than this are the same one, degrees.
SAME_DEG = 5.0
#: local: radius of the disc around the wrist pixel whose median depth is read, pixels.
LOCAL_RADIUS_PX = 6
FIXES = ("v2", "v2_relgate", "plane_pnpray", "local", "pnp_cont", "pnp_sensor", "pnp_align", "pnp_align_absonly")
VARIANTS = ("raw", "fused", "fused_rk") + tuple(v for f in FIXES for v in (f, f + "_rk"))


class Gate:
    """DepthFusion's gate and fallback on any per-frame sensor / PnP ratio (vision/depth_wrist.py)."""

    def __init__(self, relative: bool) -> None:
        self.relative = relative
        self.recent: deque[float] = deque(maxlen=HOLD_WINDOW)

    def __call__(self, s):
        held = float(np.median(self.recent)) if self.recent else None
        use = s is not None and SCALE_BAND[0] <= s <= SCALE_BAND[1]
        if use:
            if self.relative and len(self.recent) >= MIN_HOLD:
                use = abs(s / held - 1.0) <= REL_BAND
            self.recent.append(s)
        if use:
            return "depth", s
        return ("hold", held) if held is not None else ("pnp", 1.0)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise angle between rotation matrices (..., 3, 3), degrees."""
    c = (np.einsum("...ij,...ij->...", a, b) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def normal_angle_deg(rot: np.ndarray, normal: np.ndarray) -> float:
    """Angle between a palm frame's x axis (palm normal) and a plane normal, sign ignored, degrees."""
    return float(np.degrees(np.arccos(min(1.0, abs(rot[:, 0] @ normal)))))


def smallest_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a to unit vector b about their common normal."""
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(a @ b)
    return np.eye(3) if s < 1e-12 else cv2.Rodrigues(axis / s * np.arctan2(s, c))[0]


def rotate_about(hand: np.ndarray, a: np.ndarray, b: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """smallest_rotation(a, b) applied to the points about centre: (rotation, turned points)."""
    rot = smallest_rotation(a, b)
    return rot, centre + (hand - centre) @ rot.T


def plane_normal(points: np.ndarray) -> np.ndarray:
    n = np.linalg.svd(points - points.mean(axis=0))[2][-1]
    return n if n @ points.mean(axis=0) > 0 else -n


def pnp_minima(world, pixels, k_pnp, p, rot, side):
    """[(position, palm frame, rms px)] for SQPnP's solution and the LM minimum from its mirrored tilt."""
    r0 = rot @ palm_frame(world, side).T
    t0 = p - r0 @ world[0]
    hand = world @ r0.T + t0
    palm = hand[list(PALM)]
    c = palm.mean(axis=0)
    n, u = plane_normal(palm), c / np.linalg.norm(c)
    rot_f = smallest_rotation(n, 2.0 * (n @ u) * u - n)  # tilt mirrored about the line of sight
    r_f, t_f = rot_f @ r0, c + rot_f @ (t0 - c)
    out = []
    for rr, tt, refine in ((r0, t0, False), (r_f, t_f, True)):
        rv, tv = cv2.Rodrigues(rr)[0], tt.reshape(3, 1).copy()
        if refine:
            rv, tv = cv2.solvePnPRefineLM(world, pixels, k_pnp, np.zeros(5), rv, tv)
        proj = cv2.projectPoints(world, rv, tv, k_pnp, np.zeros(5))[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((proj - pixels) ** 2, axis=1))))
        placed = world @ cv2.Rodrigues(rv)[0].T + tv.ravel()
        if placed[:, 2].min() <= 0:
            continue
        out.append((placed[0], palm_frame(placed, side), rms))
    if len(out) == 2 and angle_deg(out[0][1], out[1][1]) < SAME_DEG:
        out = out[:1]
    return out


def rot_steps(rot: np.ndarray, t: np.ndarray, mask: np.ndarray, pairs: np.ndarray | None = None) -> dict:
    """Palm frame turn between consecutive frames that are both in mask (and in `pairs`, a mask over
    np.diff rows, if given), degrees: p50, p95 and the number over FLIP_DEG."""
    use = steps(t) & mask[1:] & mask[:-1]
    if pairs is not None:
        use &= pairs
    if use.sum() < 10:
        return {"n_steps": int(use.sum())}
    a = angle_deg(rot[:-1], rot[1:])[use]
    return {"n_steps": int(use.sum()), "step_p50_deg": r(np.percentile(a, 50)), "step_p95_deg": r(np.percentile(a, 95)),
            "steps_over_flip_deg": int(np.sum(a > FLIP_DEG))}


def time_v2(det, depth, depth_scale, k, side: str, idx: np.ndarray) -> dict:
    """WristDepthV2.__call__ wall time per detection, ms, depth converted to metres beforehand as the tracker would."""
    v2, ms, conv = WristDepthV2(), [], []
    # read out of the npz first: indexing det[...] inside the timed call would load the whole array
    p, rot, image, world = (det[f"{side}_{a}"] for a in ("p", "R", "image", "world"))
    for i in idx:
        raw = np.array(depth[i])
        pose, im, w = SimpleNamespace(position=p[i], rotation=rot[i]), image[i], world[i]
        t0 = time.perf_counter()
        dm = raw.astype(np.float64) * depth_scale
        t1 = time.perf_counter()
        v2(pose, im, w, dm, k, side)
        ms.append((time.perf_counter() - t1) * 1000.0)
        conv.append((t1 - t0) * 1000.0)
    return pct(np.array(ms), qs=(50, 95, 99), n=3) | {"depth_to_metres_ms": pct(np.array(conv), qs=(50, 95), n=3)}


def evaluate_side(det, depth, depth_scale, k, k_pnp, r_cb, side: str) -> dict:
    found = det[f"{side}_found"] & np.isfinite(det[f"{side}_p"]).all(axis=1)
    idx = np.flatnonzero(found)
    out: dict = {"n_found": int(len(idx))}
    if len(idx) < 10:
        return out
    t, phase = det["t"][idx], det["phase"][idx]
    p_raw, r_raw = det[f"{side}_p"][idx], det[f"{side}_R"][idx]
    world, image = det[f"{side}_world"][idx], det[f"{side}_image"][idx]
    size = depth.shape[2:0:-1]
    n = len(idx)

    pos = {v: np.empty((n, 3)) for v in ("raw", "fused") + FIXES}
    rot = {v: np.empty((n, 3, 3)) for v in ("raw", "fused") + FIXES}
    src = {v: np.empty(n, dtype=object) for v in ("fused",) + FIXES}
    sensor_normal = np.full((n, 3), np.nan)
    plane_rms = np.full(n, np.nan)
    n_minima, cont_alt, sensor_alt = np.zeros(n, int), np.zeros(n, bool), np.zeros(n, bool)
    v1 = DepthFusion()
    v2 = WristDepthV2()
    fus = {"pnp_cont": DepthFusion(), "pnp_sensor": DepthFusion(), "pnp_align": DepthFusion(),
           "pnp_align_absonly": DepthFusion(rel_band=np.inf)}
    gates = {"v2_relgate": Gate(relative=True), "plane_pnpray": Gate(relative=False), "local": Gate(relative=False)}
    prev_choice, prev_t = None, None
    for j, i in enumerate(idx):
        dm = depth[i].astype(np.float64) * depth_scale
        p, rr, w, im = p_raw[j], r_raw[j], world[j], image[j]
        pos["raw"][j], rot["raw"][j] = p, rr
        sensor = palm_depth(dm, im)
        plane = sensor_plane(sensor, k) if sensor is not None else None
        if plane is not None:
            sensor_normal[j], plane_rms[j] = plane.normal, plane.rms

        f = v1(p, rr, w, sensor, k, side)
        pos["fused"][j], rot["fused"][j], src["fused"][j] = f.position, rr, f.source

        f2 = v2(SimpleNamespace(position=p, rotation=rr), im, w, dm, k, side)
        pos["v2"][j], src["v2"][j] = f2.position, f2.source
        rot["v2"][j] = f2.rotation if f2.rotation is not None else rr
        how, sc = gates["v2_relgate"](f2.measured)
        pos["v2_relgate"][j] = f2.position if how == "depth" else p * sc
        rot["v2_relgate"][j], src["v2_relgate"][j] = (rot["v2"][j] if how == "depth" else rr), how

        # plane_pnpray: PnP's wrist bearing cut by the same plane, v2's palm frame
        cut = None if plane is None else on_plane(plane, (p[:2] / p[2]) * k[[0, 1], [0, 1]] + k[:2, 2], k)
        s = None if cut is None else float(np.linalg.norm(cut[0]) / np.linalg.norm(p))
        how, sc = gates["plane_pnpray"](s)
        pos["plane_pnpray"][j] = cut[0] if how == "depth" else p * sc
        rot["plane_pnpray"][j] = f2.rotation if how == "depth" and f2.rotation is not None else rr
        src["plane_pnpray"][j] = how

        # local: median depth of a disc around the wrist pixel, on the wrist pixel's ray
        wpx = im[0, :2] * size
        s = None
        ui, vi = int(round(wpx[0])), int(round(wpx[1]))
        rad = LOCAL_RADIUS_PX
        if rad <= ui < size[0] - rad and rad <= vi < size[1] - rad:
            yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
            disc = dm[vi + yy[xx ** 2 + yy ** 2 <= rad ** 2], ui + xx[xx ** 2 + yy ** 2 <= rad ** 2]]
            disc = disc[(disc > 0.1) & (disc < 3.0)]
            if len(disc) >= 10:
                q = pixel_rays(wpx, k)[0] * float(np.median(disc))
                s = float(np.linalg.norm(q) / np.linalg.norm(p))
        how, sc = gates["local"](s)
        pos["local"][j] = q if how == "depth" else p * sc
        rot["local"][j], src["local"][j] = rr, how

        # fix b: PnP's two minima
        mins = pnp_minima(w, im[:, :2] * size, k_pnp, p, rr, side)
        n_minima[j] = len(mins)
        best = min(m[2] for m in mins)
        near = [m for m in mins if m[2] <= best + CONT_TOL_PX]
        if prev_choice is None or t[j] - prev_t > 0.3:
            pick = min(near, key=lambda m: m[2])
        else:
            pick = min(near, key=lambda m: angle_deg(m[1], prev_choice))
        prev_choice, prev_t = pick[1], t[j]
        cont_alt[j] = pick is not mins[0]
        f = fus["pnp_cont"](pick[0], pick[1], w, sensor, k, side)
        pos["pnp_cont"][j], rot["pnp_cont"][j], src["pnp_cont"][j] = f.position, pick[1], f.source

        pick = mins[0] if plane is None else min(mins, key=lambda m: normal_angle_deg(m[1], plane.normal))
        sensor_alt[j] = pick is not mins[0]
        f = fus["pnp_sensor"](pick[0], pick[1], w, sensor, k, side)
        pos["pnp_sensor"][j], rot["pnp_sensor"][j], src["pnp_sensor"][j] = f.position, pick[1], f.source

        # fix c: PnP's hand turned so its palm normal is the sensor plane's
        pa, ra = p, rr
        if plane is not None:
            hand = hand_in_camera(p, rr, w, side)
            palm = hand[list(PALM)]
            turn, hand2 = rotate_about(hand, plane_normal(palm), plane.normal, palm.mean(axis=0))
            pa, ra = hand2[0], turn @ rr
        for v in ("pnp_align", "pnp_align_absonly"):
            f = fus[v](pa, ra, w, sensor, k, side)
            pos[v][j], rot[v][j], src[v][j] = f.position, ra, f.source

    still, free = phase == "still", phase == "free"
    has_plane = np.isfinite(sensor_normal).all(axis=1)
    out["coverage"] = {
        v: {"depth": r(np.mean(src[v] == "depth"), 4), "hold": r(np.mean(src[v] == "hold"), 4),
            "pnp": r(np.mean(src[v] == "pnp"), 4),
            "by_phase_depth": {ph: r(np.mean(src[v][phase == ph] == "depth"), 4) for ph in np.unique(phase)}}
        for v in src
    }
    out["sensor_plane"] = {
        "frames_with_plane": r(has_plane.mean(), 4),
        "rms_mm": pct(plane_rms * 1000, n=2),
        "normal_still_std": [r(x, 4) for x in np.std(sensor_normal[still & has_plane], axis=0)]
        if (still & has_plane).sum() > 1 else None,
    }
    out["pnp_minima"] = {
        "two_minima_fraction": r(np.mean(n_minima == 2), 4),
        "pnp_cont_picked_other": r(cont_alt.mean(), 4),
        "pnp_sensor_picked_other": r(sensor_alt.mean(), 4),
        "pnp_sensor_picked_other_still": r(sensor_alt[still].mean(), 4) if still.any() else None,
    }
    out["v2_call_ms"] = time_v2(det, depth, depth_scale, k, side, idx)

    series = {"raw": pos["raw"], "fused": pos["fused"]}
    unfiltered = {"raw": pos["raw"], "fused": pos["fused"]}
    rots = {"raw": rot["raw"], "fused": rot["fused"]}
    for v in ("fused",) + FIXES:
        if v != "fused":
            series[v], unfiltered[v], rots[v] = pos[v], pos[v], rot[v]
        sig = np.where(src[v] == "depth", SIGMA_RAY_FUSED, DEFAULT_SIGMA_RAY)
        series[v + "_rk"], unfiltered[v + "_rk"], rots[v + "_rk"] = run_kalman(pos[v], t, sig), pos[v], rot[v]
    r_med = float(np.median(np.linalg.norm(p_raw, axis=1)))
    bearing = lambda q: q / np.linalg.norm(q, axis=1, keepdims=True) * r_med  # noqa: E731
    normal_ang = {v: np.array([normal_angle_deg(rots[v][j], sensor_normal[j]) if has_plane[j] else np.nan
                               for j in range(n)]) for v in ("raw",) + FIXES}
    out["variants"] = {}
    for v in VARIANTS:
        q = series[v]
        out["variants"][v] = {
            "still": still_jitter(q, t, still),
            "lag_lateral_vs_raw_ms": lag_ms(t, bearing(p_raw), bearing(q)),
            "lag_ray_vs_unfiltered_ms": lag_ms(t, np.linalg.norm(unfiltered[v], axis=1), np.linalg.norm(q, axis=1)),
            "free_robot": robot_steps(q, t, free, r_cb),
            "rotation_still": rot_steps(rots[v], t, still),
            "rotation_free": rot_steps(rots[v], t, free),
        }
        base = v[:-3] if v.endswith("_rk") else v
        if base in src:
            both = (src[base][1:] == "depth") & (src[base][:-1] == "depth")
            out["variants"][v]["rotation_free_sensor_frames"] = rot_steps(rots[v], t, free, both)
        if base in normal_ang and not v.endswith("_rk"):
            out["variants"][v]["normal_vs_sensor_deg"] = {"still": pct(normal_ang[base][still], n=1),
                                                          "all": pct(normal_ang[base], n=1)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="clip names under data/clips/")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    clips: dict = {}
    k_pnp = camera.load(REPO / "config" / "ego_camera_info.yaml").camera_matrix  # what the detections were solved with
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
        phases, counts = np.unique(det["phase"], return_counts=True)
        clips[name] = {
            "n_frames": int(len(det["t"])),
            "phase_frames": {str(ph): int(cn) for ph, cn in zip(phases, counts)},
            "depth_stream": meta.get("depth_stream"),
            "sides": {s: evaluate_side(det, depth, meta["depth_scale"], k, k_pnp, r_cb, s) for s in SIDES},
        }
        for s in SIDES:
            res = clips[name]["sides"][s]
            if "variants" not in res:
                print(f"{name:14s} {s:5s} found {res['n_found']}: too few")
                continue
            print(f"{name:14s} {s:5s} found {res['n_found']}  minima {res['pnp_minima']}  v2 call {res['v2_call_ms']}")
            for v in VARIANTS:
                m = res["variants"][v]
                cov = res["coverage"].get(v[:-3] if v.endswith("_rk") else v, {}).get("depth")
                st = m["still"]
                print(f"  {v:15s} cov {cov}  still ray {st.get('std_ray_mm')}/{st.get('step_p95_ray_mm')}  "
                      f"lat {st.get('std_lat_mm')}/{st.get('step_p95_lat_mm')}  lag {m['lag_lateral_vs_raw_ms']}/"
                      f"{m['lag_ray_vs_unfiltered_ms']}  free {[m['free_robot'].get(f'step_p95_{a}_mm') for a in 'xyz']}  "
                      f"rot still {m['rotation_still'].get('step_p95_deg')} flips {m['rotation_still'].get('steps_over_flip_deg')}"
                      f"  rot free {m['rotation_free'].get('step_p95_deg')}"
                      f"  n-ang {m.get('normal_vs_sensor_deg', {}).get('still', {}).get('p50')}")

    payload = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "tool": "tools/eval_depth_wrist_v2.py",
            "detections": "data/results/m5_detections_<clip>.npz (tools/extract_ego_detections.py)",
            "metrics": "tools/eval_depth_wrist.py (M5) still_jitter, lag_ms, robot_steps, imported",
            "intrinsics": "rays: meta.json colour intrinsics; PnP minima: config/ego_camera_info.yaml",
            "gate": {"abs_band": list(SCALE_BAND), "rel_band": REL_BAND, "min_hold": MIN_HOLD, "hold_window": HOLD_WINDOW,
                     "ratio": "fused/pnp_*: depth_wrist.depth_scale; v2/plane_pnpray/local: sensor wrist distance / PnP wrist distance"},
            "ray_kalman_rk": {"sigma_ray_on_depth_frames_m": SIGMA_RAY_FUSED, "sigma_ray_otherwise_m": DEFAULT_SIGMA_RAY},
            "flip_deg": FLIP_DEG, "cont_tol_px": CONT_TOL_PX, "same_minimum_deg": SAME_DEG,
            "local_radius_px": LOCAL_RADIUS_PX, "min_cos": MIN_COS,
            "rotation": "palm frame (vision.wrist_pose.palm_frame); raw/fused/local: PnP's; v2/plane_pnpray: from the sensor plane on depth frames, PnP's otherwise; *_rk: same as unfiltered",
            "rotation_step": "angle of R_prev^T R between consecutive frames (< 0.05 s apart), degrees",
            "consecutive": f"frame pairs less than {MAX_STEP_S} s apart",
            "v2_call_ms": "WristDepthV2.__call__ wall time per hand, depth already in metres (float64 640x480); "
                          "depth_to_metres_ms: the uint16 -> float64 metres conversion the caller does once per frame",
            "cpu": "measured on the lead's workstation while other agents may have been running",
        },
        "clips": clips,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
