#!/usr/bin/env python3
"""
eval_depth_wrist_v4_cross.py — milestone M9: where fusion v4 stops working, across all four depth clips.

tools/eval_depth_wrist_v4.py (M8) compared v3 and v4 on ego_depth_424 only, the clip v4's
gain was chosen on. This runs the same gap simulation (spans of 0.3 / 1 / 2 s with
depth_m=None, scenarios clean / depth_phase / contaminated, reference = v2 on the same
frames without the gap) on every clip given, and adds:

  * a gain sweep: WristDepthV4(gain=g) for g in GAINS, with the default limits; g = 0.5 is
    v4's default and g = 0 should be v3 (checked frame by frame: max_abs_diff_g0_vs_v3_mm);
  * leave-one-clip-out gain choice on the depth_phase 2 s gaps;
  * per-gap records with the palm's tilt during the gap: the angle between the sensor
    palm-plane normal at a gap frame (or the first frame after it) and the one on the last
    sensor frame before the gap (the anchor), and the same from MediaPipe world landmarks;
    plus the span change, the reference's true range change, finger flexion change and
    frame-to-frame span jitter;
  * natural gaps: runs of v3 `hold` frames in the unmasked clip, with the fusion reset
    after more than RESET_GAP_S without a detection as the live tracker does.

MediaPipe is not run again (data/results/m5_detections_<clip>.npz); sensor palm planes are
computed once per detection as in M8. Fusion classes are used unchanged; only measure()
returns the cached value (M8's CachedV2 / CachedV3 / CachedV4, imported).

Usage (from the repository root, .venv-vision or .venv-arm):
    .venv-vision/bin/python -B tools/eval_depth_wrist_v4_cross.py [clip ...]
Writes data/results/m9_v4_cross.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import eval_depth_wrist_v4 as m8  # noqa: E402
from vision.depth_filter import DEFAULT_SIGMA_RAY, RESET_GAP_S, palm_frame  # noqa: E402
from vision.depth_filter_v2 import RayKalmanV2  # noqa: E402
from vision.depth_wrist_v2 import pixel_rays  # noqa: E402
from vision.depth_wrist_v3 import WristDepthV3  # noqa: E402
from vision.depth_wrist_v4 import GAIN, MAX_REL, STEP, hand_span  # noqa: E402

OUT = REPO / "data" / "results" / "m9_v4_cross.json"
CLIPS = ("ego_depth", "ego_depth_424", "ego_depth_work", "smoke")
SIDES = ("left", "right")
SCENARIOS = ("clean", "depth_phase", "contaminated")
GAINS = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANTS = ("v3",) + tuple(f"g{g:g}" for g in GAINS)  # g0.5 is v4 with its defaults
V4 = f"g{GAIN:g}"
ANGLE_BINS = ((0.0, 15.0), (15.0, 30.0), (30.0, 180.0))
TIPS = (4, 8, 12, 16, 20)
#: thresholds of the failure-cause labels (see classify())
ROT_DEG, FLEX_LOG, JITTER_LOG = 20.0, 0.10, 0.10
#: a gap whose reference range changed by less than this (|log|) counts as "no motion along the ray"
STILL_LOG = 0.05
#: anchor range bins (m): the hand near the operator vs a hand far from the camera (smoke's other person)
RANGE_BINS = ((0.0, 0.5), (0.5, 9.0))
#: scope "near": only gaps whose anchor range is below this (drops smoke's far, other person's hand)
NEAR_M = 0.5
stats, r = m8.stats, m8.r


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def load(clip: str) -> dict:
    cdir = REPO / "data" / "clips" / clip
    det = np.load(REPO / "data" / "results" / f"m5_detections_{clip}.npz")
    meta = json.loads((cdir / "meta.json").read_text())
    depth = np.load(cdir / "depth.npy", mmap_mode="r")
    if len(depth) != len(det["t"]):
        raise SystemExit(f"{clip}: depth.npy has {len(depth)} rows, detections {len(det['t'])}")
    c = meta["color"]
    k = np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])
    size = (depth.shape[2], depth.shape[1])
    fps = 1.0 / float(np.median(np.diff(det["t"])))
    dummy = np.zeros(depth.shape[1:])
    stream = meta.get("depth_stream", {"width": 640, "height": 480, "note": "not in meta.json; 640x480 per M5"})
    sides = {}
    for side in SIDES:
        idx = np.flatnonzero(det[f"{side}_found"] & np.isfinite(det[f"{side}_p"]).all(axis=1))
        image = det[f"{side}_image"][idx]
        world = det[f"{side}_world"][idx]
        measure = WristDepthV3().measure
        cache = [measure(image[j], depth[i].astype(np.float64) * meta["depth_scale"], k) for j, i in enumerate(idx)]
        n_sensor = np.full((len(idx), 3), np.nan)
        for j, (hand, _) in enumerate(cache):
            if hand is not None:
                n_sensor[j] = palm_frame(hand, side)[:, 0]
        n_world = np.array([palm_frame(w, side)[:, 0] for w in world]) if len(idx) else np.zeros((0, 3))
        flex = np.array([np.mean(np.linalg.norm(w[list(TIPS)] - w[0], axis=1)) for w in world])
        span = np.array([hand_span(im, size) for im in image])
        wrist_px = image[:, 0, :2] * size
        rays = pixel_rays(wrist_px, k) if len(idx) else np.zeros((0, 3))
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        # hand size from world landmarks (wrist to middle MCP), a weak hint of who the hand belongs to
        hand_len = np.linalg.norm(world[:, 9] - world[:, 0], axis=1)
        sides[side] = {"side": side, "t": det["t"][idx], "phase": det["phase"][idx], "p": det[f"{side}_p"][idx],
                       "R": det[f"{side}_R"][idx], "image": image, "world": world, "cache": cache, "k": k,
                       "dummy": dummy, "n_sensor": n_sensor, "n_world": n_world, "flex": flex, "span": span,
                       "ray": rays, "hand_len": hand_len}
    t_ms = [json.loads(line)["t_color_ms"] for line in (cdir / "frames.jsonl").read_text().splitlines()[:1]]
    info = {"depth_stream": stream, "color_fx": c["fx"], "color_coeffs": c["coeffs"], "R_CB": meta.get("R_CB"),
            "first_frame_utc": datetime.datetime.fromtimestamp(t_ms[0] / 1000, datetime.timezone.utc).isoformat(timespec="seconds"),
            "frames": int(len(det["t"])), "fps": round(fps, 2)}
    return {"sides": sides, "info": info}


def make(v: str, frame):
    if v == "v2":
        return m8.CachedV2(frame)
    if v == "v3":
        return m8.CachedV3(frame)
    return m8.CachedV4(frame, gain=float(v[1:]))


def run(v: str, d: dict, frames: np.ndarray, drop: np.ndarray, pnp_scale: np.ndarray, reset: bool = False):
    """As M8's run(); with reset=True the fusion and filter are reset after > RESET_GAP_S without a detection."""
    frame = m8.Frame(d["cache"])
    fusion, kf = make(v, frame), RayKalmanV2()
    n = len(frames)
    pos, filt, src = np.empty((n, 3)), np.empty((n, 3)), np.empty(n, dtype=object)
    last = None
    for a, j in enumerate(frames):
        frame.j = j
        if reset and last is not None and d["t"][j] - last > RESET_GAP_S:
            fusion.reset()
            kf = RayKalmanV2()
        last = d["t"][j]
        pose = SimpleNamespace(position=d["p"][j] * pnp_scale[a], rotation=d["R"][j])
        f = fusion(pose, d["image"][j], d["world"][j], None if drop[a] else d["dummy"], d["k"], d["side"])
        pos[a], src[a] = f.position, f.source
        filt[a] = kf(f.position, d["t"][j], m8.SIGMA_RAY_SENSOR if f.source == "depth" else DEFAULT_SIGMA_RAY)
    return pos, src, filt


def gap_geometry(d: dict, frames: np.ndarray, g0: int, e: int, sensor_gap: np.ndarray) -> dict:
    """Tilt, span, flexion and jitter of one gap; indices are into `frames`, g0 first gap frame, e first after."""
    j_a, j_e = frames[g0 - 1], frames[e]
    ns, nw = d["n_sensor"], d["n_world"]
    sens = [frames[a] for a in np.flatnonzero(sensor_gap)] + [j_e]
    ang_s = [angle_deg(ns[j], ns[j_a]) for j in sens]
    ang_s = [a for a in ang_s if np.isfinite(a)]
    ang_w = [angle_deg(nw[j], nw[j_a]) for j in frames[g0:e + 1]]
    span = d["span"][frames[g0 - 1:e]]
    ls = np.log(span / span[0])
    cos_a, cos_e = abs(ns[j_a] @ d["ray"][j_a]), abs(ns[j_e] @ d["ray"][j_e])
    return {"angle_sensor_max_deg": r(max(ang_s)) if ang_s else None,
            "angle_sensor_end_deg": r(angle_deg(ns[j_e], ns[j_a])),
            "angle_world_max_deg": r(max(ang_w)), "angle_world_end_deg": r(ang_w[-1]),
            "span_log_end": r(ls[-1], 3), "span_log_maxabs": r(np.max(np.abs(ls)), 3),
            "span_jitter_log": r(np.max(np.abs(np.diff(np.log(d["span"][frames[g0 - 1:e + 1]])))), 3),
            "flex_log_end": r(np.log(d["flex"][frames[e - 1]] / d["flex"][j_a]), 3),
            "cos_view_ratio_end": r(cos_e / cos_a, 3) if np.isfinite(cos_a * cos_e) and cos_a > 0 else None}


def gaps(d: dict, gap_s: float, scenario: str, gmax: list) -> tuple[list, dict]:
    t, phase = d["t"], d["phase"]
    records, raw = [], {v: {"err": [], "angle": [], "angle_w": []} for v in VARIANTS}
    for start in np.arange(t[0] + m8.WARM_S, t[-1] - gap_s - m8.END_S, m8.STEP_S) if len(t) else []:
        frames = np.flatnonzero((t >= start - m8.WARM_S) & (t < start + gap_s + m8.END_S))
        tw = t[frames]
        gap = (tw >= start) & (tw < start + gap_s)
        if gap.sum() < 3:
            continue
        if scenario == "depth_phase" and not np.all(phase[frames][gap] == "depth"):
            continue
        g0, e = np.flatnonzero(gap)[0], np.flatnonzero(gap)[-1] + 1
        if g0 == 0 or e >= len(frames) or tw[e] - tw[e - 1] > m8.MAX_STEP_S or tw[g0] - tw[g0 - 1] > m8.MAX_STEP_S:
            continue
        one, nodrop = np.ones(len(frames)), np.zeros(len(frames), bool)
        ref, ref_src, ref_rk = run("v2", d, frames, nodrop, one)
        if ref_src[e] != "depth" or ref_src[g0 - 1] != "depth":
            continue
        scale = one.copy()
        if scenario == "contaminated":
            scale[(tw >= start - m8.BURST_S) & (tw < start)] = m8.BURST_SCALE
        after = (tw >= tw[e]) & (tw < tw[e] + m8.END_S)
        sensor_gap = gap & (ref_src == "depth")
        rec = {"t0": r(tw[g0], 2), "gap_s": gap_s, "n_frames": int(gap.sum()), "n_sensor": int(sensor_gap.sum()),
               "phase": str(phase[frames][g0]),
               "ref_end_step_mm": r(np.linalg.norm(ref[e] - ref[e - 1]) * 1000),
               "range_log_true_end": r(np.log(np.linalg.norm(ref[e]) / np.linalg.norm(ref[g0 - 1])), 3),
               "anchor_range_m": r(np.linalg.norm(ref[g0 - 1]), 3), "anchor_span_px": r(d["span"][frames[g0 - 1]]),
               "err_mean_mm": {}, "err_max_mm": {}, "end_step_mm": {}, "rk_max_step_after_mm": {}}
        rec.update(gap_geometry(d, frames, g0, e, sensor_gap))
        ang = np.array([angle_deg(d["n_sensor"][frames[a]], d["n_sensor"][frames[g0 - 1]]) for a in np.flatnonzero(sensor_gap)])
        ang_w = np.array([angle_deg(d["n_world"][frames[a]], d["n_world"][frames[g0 - 1]]) for a in np.flatnonzero(sensor_gap)])
        pos_v = {}
        for v in VARIANTS:
            pos, src, rk = run(v, d, frames, gap, scale)
            pos_v[v] = pos
            err = np.linalg.norm(pos[sensor_gap] - ref[sensor_gap], axis=1) * 1000
            raw[v]["err"].extend(err)
            raw[v]["angle"].extend(ang)
            raw[v]["angle_w"].extend(ang_w)
            rec["err_mean_mm"][v] = r(err.mean()) if len(err) else None
            rec["err_max_mm"][v] = r(err.max()) if len(err) else None
            rec["end_step_mm"][v] = r(np.linalg.norm(pos[e] - pos[e - 1]) * 1000)
            rec["rk_max_step_after_mm"][v] = r(np.max(np.linalg.norm(np.diff(rk[e - 1:][: after.sum() + 1], axis=0), axis=1)) * 1000)
        gmax[0] = max(gmax[0], float(np.max(np.linalg.norm(pos_v["g0"] - pos_v["v3"], axis=1))) * 1000)
        rec["held_log_v4_vs_v3_end"] = r(np.log(np.linalg.norm(pos_v[V4][e - 1]) / np.linalg.norm(pos_v["v3"][e - 1])), 3)
        records.append(rec)
    return records, raw


def summarise(records: list, raw: dict) -> dict:
    out = {"n_gaps": len(records), "reference_end_step_mm": stats([x["ref_end_step_mm"] for x in records]), "variants": {}}
    for v in VARIANTS:
        out["variants"][v] = {"gap_error_mm": stats(raw[v]["err"]),
                              "end_step_mm": stats([x["end_step_mm"][v] for x in records]),
                              "rk_max_step_after_mm": stats([x["rk_max_step_after_mm"][v] for x in records])}
    diff = [x["err_mean_mm"][V4] - x["err_mean_mm"]["v3"] for x in records if x["err_mean_mm"]["v3"] is not None]
    out["v4_minus_v3_gap_mean_error_mm"] = stats(diff)
    out["share_gaps_v4_worse_than_v3"] = r(np.mean(np.array(diff) > 0), 3) if diff else None
    return out


def natural(d: dict) -> dict:
    """Runs of v3 `hold` frames in the unmasked clip (fusion reset as the live tracker resets it)."""
    n = len(d["t"])
    if not n:
        return {"n_found": 0}
    frames, one, nodrop = np.arange(n), np.ones(n), np.zeros(n, bool)
    res = {v: run(v, d, frames, nodrop, one, reset=True) for v in ("v2", "v3", V4)}
    src = res["v3"][1]
    t = d["t"]
    runs, a = [], 0
    while a < n:
        if src[a] != "hold":
            a += 1
            continue
        b = a
        while b + 1 < n and src[b + 1] == "hold":
            b += 1
        before, back = a - 1, b + 1
        ok = (before >= 0 and back < n and src[before] == "depth" and src[back] == "depth"
              and t[a] - t[before] < m8.MAX_STEP_S and t[back] - t[b] < m8.MAX_STEP_S)
        if ok:
            ns = d["n_sensor"]
            x = {"t0": r(t[a], 2), "length_s": r(t[back] - t[a], 3), "n_frames": int(back - a),
                 "range_log_true": r(np.log(np.linalg.norm(res["v3"][0][back]) / np.linalg.norm(res["v3"][0][before])), 3),
                 "span_log": r(np.log(d["span"][b] / d["span"][before]), 3),
                 "angle_sensor_deg": r(angle_deg(ns[back], ns[before])),
                 "angle_world_max_deg": r(max(angle_deg(d["n_world"][j], d["n_world"][before]) for j in range(a, back + 1))),
                 "step_mm": {v: r(np.linalg.norm(res[v][0][back] - res[v][0][b]) * 1000) for v in res}}
            runs.append(x)
        a = b + 1
    out = {"by_source_v3": {k: int(c) for k, c in zip(*np.unique(src.astype(str), return_counts=True))},
           "n_runs": len(runs), "length_s": stats([x["length_s"] for x in runs], 2),
           "hold_to_depth_step_mm": {v: stats([x["step_mm"][v] for x in runs]) for v in res}}
    if runs:
        dv = np.array([x["step_mm"][V4] - x["step_mm"]["v3"] for x in runs])
        out["v4_minus_v3_step_mm"] = stats(dv)
        out["share_v4_worse"] = r(np.mean(dv > 0), 3)
        for lo, hi in ((0.0, 0.3), (0.3, 1.0), (1.0, 1e9)):
            sel = [x for x in runs if lo <= x["length_s"] < hi]
            out[f"steps_length_{lo:g}_{hi:g}s"] = {"n": len(sel), **{v: stats([x["step_mm"][v] for x in sel]) for v in ("v3", V4)}}
    out["runs"] = runs
    return out


def rotation_bins(items: list, key: str, bins=ANGLE_BINS, top=180.0) -> dict:
    """Per-gap bins by `key` (an angle), with pooled per-frame errors of v3 and v4. items: (record, raw errors)."""
    out = {}
    for lo, hi in bins:
        sel = [(x, e) for x, e in items if x[key] is not None and lo <= x[key] < hi]
        name = f"{lo:g}-{hi:g}" if hi < top else f">={lo:g}"
        b = {"n_gaps": len(sel)}
        for v in ("v3", V4, "g1"):
            b[v] = {"gap_error_mm": stats(np.concatenate([e[v] for _, e in sel]) if sel else []),
                    "end_step_mm": stats([x["end_step_mm"][v] for x, _ in sel])}
        diff = [x["err_mean_mm"][V4] - x["err_mean_mm"]["v3"] for x, _ in sel if x["err_mean_mm"]["v3"] is not None]
        b["v4_minus_v3_gap_mean_error_mm"] = stats(diff)
        b["share_gaps_v4_worse"] = r(np.mean(np.array(diff) > 0), 3) if diff else None
        de = [x["end_step_mm"][V4] - x["end_step_mm"]["v3"] for x, _ in sel]
        b["v4_minus_v3_end_step_mm"] = stats(de)
        b["share_gaps_v4_end_step_worse"] = r(np.mean(np.array(de) > 0), 3) if de else None
        out[name] = b
    return out


def frame_bins(raw_list: list, key: str) -> dict:
    """Per-frame bins by the frame's own tilt from the anchor (sensor or world normal)."""
    out = {}
    ang = np.concatenate([np.asarray(rw["v3"][key], float) for rw in raw_list]) if raw_list else np.zeros(0)
    errs = {v: np.concatenate([np.asarray(rw[v]["err"], float) for rw in raw_list]) if raw_list else np.zeros(0)
            for v in ("v3", V4, "g1")}
    for lo, hi in ANGLE_BINS:
        m = np.isfinite(ang) & (ang >= lo) & (ang < hi)
        name = f"{lo:g}-{hi:g}" if hi < 180 else f">={lo:g}"
        out[name] = {"n_frames": int(m.sum()), **{v: stats(errs[v][m]) for v in errs},
                     "share_frames_v4_worse": r(np.mean(errs[V4][m] > errs["v3"][m]), 3) if m.any() else None}
    return out


def classify(x: dict) -> list:
    lab = []
    a = max([v for v in (x["angle_sensor_max_deg"], x["angle_sensor_end_deg"]) if v is not None], default=None)
    if a is not None and a >= ROT_DEG:
        lab.append("rotation")
    if x["flex_log_end"] is not None and abs(x["flex_log_end"]) >= FLEX_LOG:
        lab.append("finger_flexion")
    if x["span_jitter_log"] is not None and x["span_jitter_log"] >= JITTER_LOG:
        lab.append("detection_jitter")
    return lab or ["other"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", default=list(CLIPS))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--gaps", default=",".join(f"{g:g}" for g in m8.GAP_S))
    args = ap.parse_args()
    scenarios, gap_list = args.scenarios.split(","), [float(g) for g in args.gaps.split(",")]

    gmax = [0.0]
    clips, keep_raw = {}, {}
    for clip in args.clips:
        L = load(clip)
        cres = {"info": L["info"], "sides": {}}
        for side in SIDES:
            d = L["sides"][side]
            dist = [np.linalg.norm(h[0]) for h, _ in d["cache"] if h is not None]
            sres = {"n_found": int(len(d["t"])),
                    "sensor_wrist_distance_m": {f"p{q}": r(np.percentile(dist, q), 3) for q in (5, 50, 95)} if dist else None,
                    "world_wrist_to_middle_mcp_mm": stats(d["hand_len"] * 1000) if len(d["t"]) else None,
                    "natural": natural(d), "gaps": {}}
            for sc in scenarios:
                for g in gap_list:
                    key = f"{sc}_{g:g}s"
                    recs, raw = gaps(d, g, sc, gmax)
                    keep_raw[(clip, side, key)] = (recs, raw)
                    sres["gaps"][key] = {**summarise(recs, raw), "records": recs}
                    s = sres["gaps"][key]
                    print(f"{clip:15s} {side:5s} {key:18s} n {s['n_gaps']:3d}  "
                          + "  ".join(f"{v} err {s['variants'][v]['gap_error_mm'].get('p50')}/{s['variants'][v]['gap_error_mm'].get('p95')}"
                                      f" end {s['variants'][v]['end_step_mm'].get('p50')}/{s['variants'][v]['end_step_mm'].get('p95')}"
                                      for v in ("v3", V4)), flush=True)
            nat = sres["natural"]
            print(f"{clip:15s} {side:5s} natural runs {nat.get('n_runs')} {nat.get('hold_to_depth_step_mm')}", flush=True)
            cres["sides"][side] = sres
        clips[clip] = cres

    # pooled (both sides) per clip and gain: the gain sweep and leave-one-clip-out choice
    def pooled(clip_list, key, v):
        err, end = [], []
        for c in clip_list:
            for side in SIDES:
                if (c, side, key) in keep_raw:
                    recs, raw = keep_raw[(c, side, key)]
                    err.extend(raw[v]["err"])
                    end.extend(x["end_step_mm"][v] for x in recs)
        return {"gap_error_mm": stats(err), "end_step_mm": stats(end)}

    metrics = (("end_step_mm", "p50"), ("end_step_mm", "p95"), ("gap_error_mm", "p50"), ("gap_error_mm", "p95"))
    sweep, loo = {}, {}
    for key in [k for k in (f"depth_phase_{gap_list[-1]:g}s", f"clean_{gap_list[-1]:g}s") if any(k == kk for (_, _, kk) in keep_raw)]:
        sweep[key] = {c: {v: pooled([c], key, v) for v in VARIANTS} for c in args.clips}
        loo[key] = {}
        for c in args.clips:
            others = [o for o in args.clips if o != c]
            res = {}
            for mname, q in metrics:
                train = {v: pooled(others, key, v)[mname].get(q) for v in VARIANTS if v != "v3"}
                train = {v: x for v, x in train.items() if x is not None}
                test = {v: sweep[key][c][v][mname].get(q) for v in VARIANTS}
                test_g = {v: x for v, x in test.items() if v != "v3" and x is not None}
                chosen = min(train, key=train.get) if train else None
                best = min(test_g, key=test_g.get) if test_g else None
                res[f"{mname}_{q}"] = {"chosen_on_others": chosen, "train_value": train.get(chosen) if chosen else None,
                                       "held_out_at_chosen": test.get(chosen) if chosen else None,
                                       "held_out_at_0.5": test.get(V4), "held_out_v3": test.get("v3"),
                                       "held_out_best_gain": best, "held_out_best_value": test_g.get(best) if best else None,
                                       "n_gaps_held_out": sweep[key][c]["v3"]["end_step_mm"].get("n", 0)}
            loo[key][c] = res

    # rotation bins (clean scenario), per gap length, all clips pooled and per clip
    rot = {}
    for g in gap_list:
        key = f"clean_{g:g}s"
        if not any(k == key for (_, _, k) in keep_raw):
            continue
        rot[key] = {}
        for scope in ["all", "near"] + list(args.clips):
            items, raws = [], []
            for (c, side, k), (recs, raw) in keep_raw.items():
                if k != key or (scope not in ("all", "near") and c != scope):
                    continue
                # split the pooled raw errors back into gaps (records are in order, n_sensor frames each)
                o, kept = 0, {v: {"err": [], "angle": [], "angle_w": []} for v in VARIANTS}
                for x in recs:
                    n = x["n_sensor"]
                    if scope != "near" or x["anchor_range_m"] < NEAR_M:
                        items.append((x, {v: np.asarray(raw[v]["err"][o:o + n]) for v in VARIANTS}))
                        for v in VARIANTS:
                            for f in ("err", "angle", "angle_w"):
                                kept[v][f].extend(raw[v][f][o:o + n])
                    o += n
                raws.append(kept)
            rot[key][scope] = {"by_gap_sensor_max_angle": rotation_bins(items, "angle_sensor_max_deg"),
                               "by_gap_world_max_angle": rotation_bins(items, "angle_world_max_deg"),
                               "by_gap_sensor_max_angle_range_still": rotation_bins(
                                   [z for z in items if abs(z[0]["range_log_true_end"]) < STILL_LOG], "angle_sensor_max_deg"),
                               "by_gap_world_max_angle_range_still": rotation_bins(
                                   [z for z in items if abs(z[0]["range_log_true_end"]) < STILL_LOG], "angle_world_max_deg"),
                               "by_anchor_range_m": rotation_bins(items, "anchor_range_m", RANGE_BINS, 9.0),
                               "by_frame_sensor_angle": frame_bins(raws, "angle"),
                               "by_frame_world_angle": frame_bins(raws, "angle_w")}
            if scope == "all":
                a_s = np.array([x["angle_sensor_end_deg"] for x, _ in items if x["angle_sensor_end_deg"] is not None
                                and x["angle_world_end_deg"] is not None], float)
                a_w = np.array([x["angle_world_end_deg"] for x, _ in items if x["angle_sensor_end_deg"] is not None
                                and x["angle_world_end_deg"] is not None], float)
                rot[key]["sensor_vs_world_end_angle"] = {"n": int(len(a_s)), "pearson_r": r(np.corrcoef(a_s, a_w)[0, 1], 3)
                                                         if len(a_s) > 2 else None,
                                                         "median_abs_diff_deg": r(np.median(np.abs(a_s - a_w))) if len(a_s) else None}

    # span vs true range: the gain that would have been right (OLS through 0 of true log range on -log span), depth_phase longest
    ideal = {}
    key = f"depth_phase_{gap_list[-1]:g}s"
    for c in args.clips:
        xs, ys = [], []
        for side in SIDES:
            for x in keep_raw.get((c, side, key), ([], None))[0]:
                xs.append(-x["span_log_end"])
                ys.append(x["range_log_true_end"])
        xs, ys = np.array(xs, float), np.array(ys, float)
        ideal[c] = {"n": int(len(xs)), "slope_through_0": r(float(xs @ ys / (xs @ xs)), 3) if len(xs) and xs @ xs > 0 else None,
                    "pearson_r": r(np.corrcoef(xs, ys)[0, 1], 3) if len(xs) > 2 else None}

    # failure cases: clean gaps where v4's mean gap error exceeds v3's the most, non-overlapping
    cand = []
    for (c, side, k), (recs, _) in keep_raw.items():
        if not k.startswith("clean_"):
            continue
        for x in recs:
            if x["err_mean_mm"]["v3"] is None:
                continue
            cand.append((x["err_mean_mm"][V4] - x["err_mean_mm"]["v3"], c, side, x))
    cand.sort(key=lambda z: -z[0])

    def top(only_near: bool, n: int = 10) -> list:
        out = []
        for dv, c, side, x in cand:
            if only_near and x["anchor_range_m"] >= NEAR_M:
                continue
            if any(f["clip"] == c and f["side"] == side and x["t0"] < f["t0"] + f["gap_s"] and f["t0"] < x["t0"] + x["gap_s"]
                   for f in out):
                continue
            out.append({"clip": c, "side": side, "t0": x["t0"], "gap_s": x["gap_s"], "phase": x["phase"],
                        "v4_minus_v3_gap_mean_error_mm": r(dv), "err_mean_mm": {v: x["err_mean_mm"][v] for v in ("v3", V4)},
                        "end_step_mm": {v: x["end_step_mm"][v] for v in ("v3", V4)},
                        **{f: x[f] for f in ("span_log_end", "span_log_maxabs", "range_log_true_end", "held_log_v4_vs_v3_end",
                                             "angle_sensor_max_deg", "angle_sensor_end_deg", "angle_world_max_deg",
                                             "flex_log_end", "span_jitter_log", "cos_view_ratio_end", "anchor_range_m",
                                             "anchor_span_px")},
                        "causes": classify(x)})
            if len(out) == n:
                break
        return out

    fails, fails_near = top(False), top(True)
    n_cand = len(cand)
    share_worse = r(np.mean([z[0] > 0 for z in cand]), 3) if cand else None

    payload = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "tool": "tools/eval_depth_wrist_v4_cross.py",
            "clips": list(args.clips),
            "detections": "data/results/m5_detections_<clip>.npz",
            "simulation": "as tools/eval_depth_wrist_v4.py (M8): spans, step, warm-up, scenarios, reference v2 without the gap",
            "gap_s": gap_list, "scenarios": scenarios,
            "variants": {"v3": "WristDepthV3", **{f"g{g:g}": f"WristDepthV4(gain={g:g}), max_rel {MAX_REL}, step {STEP}"
                                                  for g in GAINS}},
            "v4_default": V4,
            "tilt": "angle between palm normals (palm_frame x axis) at a frame and at the anchor (last sensor frame "
                    "before the gap). sensor: palm_frame of the landmarks on the sensor palm plane (only frames with a "
                    "plane); world: palm_frame of MediaPipe world landmarks. Per-gap: max over sensor gap frames and "
                    "the first frame after; end: first frame after the gap",
            "range_still": f"*_range_still bins: only gaps with |range_log_true_end| < {STILL_LOG}",
            "anchor_range_m": "|reference wrist| on the anchor frame; by_anchor_range_m bins it (m)",
            "span_log_end": "log(span at the last gap frame / span at the anchor frame), raw spans (not v4's 3-frame median)",
            "range_log_true_end": "log(|reference wrist| first frame after / last frame before the gap)",
            "held_log_v4_vs_v3_end": "log(|v4 wrist| / |v3 wrist|) at the last gap frame",
            "flex_log_end": "log(mean world-landmark fingertip-to-wrist distance, last gap frame / anchor)",
            "span_jitter_log": "max |log span ratio| between consecutive frames from the anchor to the first frame after",
            "cos_view_ratio_end": "|sensor normal . wrist ray| first frame after / anchor (foreshortening of the palm)",
            "failure_causes": {"rotation": f"sensor tilt (max or end) >= {ROT_DEG} deg",
                               "finger_flexion": f"|flex_log_end| >= {FLEX_LOG}",
                               "detection_jitter": f"span_jitter_log >= {JITTER_LOG}", "other": "none of the above"},
            "natural": f"runs of v3 hold frames between two sensor frames (< {m8.MAX_STEP_S} s apart at both ends), "
                       f"fusion and filter reset after > {RESET_GAP_S} s without a detection (as hand_tracker_ego_v3)",
            "near": f"rotation scope 'near' and failures top_near: only gaps with anchor_range_m < {NEAR_M}",
            "loo": "for each clip, the gain (from GAINS, v3 excluded) with the lowest metric on the other clips' "
                   "pooled gaps (both hands), then that metric on the held-out clip",
            "ideal_gain": "OLS slope through 0 of range_log_true_end on -span_log_end, depth_phase longest gaps",
        },
        "max_abs_diff_g0_vs_v3_mm": r(gmax[0], 6),
        "clips": clips,
        "gain_sweep_pooled_sides": sweep,
        "leave_one_clip_out": loo,
        "ideal_gain": ideal,
        "rotation": rot,
        "failures": {"n_clean_gaps": n_cand, "share_v4_worse": share_worse, "top": fails, "top_near": fails_near},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"g0 vs v3 max diff {gmax[0]:.6f} mm -> {args.out}")


if __name__ == "__main__":
    main()
