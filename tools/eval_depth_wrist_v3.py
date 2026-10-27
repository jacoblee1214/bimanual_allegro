#!/usr/bin/env python3
"""
eval_depth_wrist_v3.py — milestone M7: what the wrist does while the sensor gives no palm depth, offline.

vision/depth_wrist_v2.py (WristDepthV2) holds PnP's wrist times the median of the last
30 in-band ratios sensor distance / PnP distance when the sensor drops out. On the live
dry run before R6 that median was contaminated by a short PnP error just before a gap
(docs/results/r6_ego_depth_follow.md). ego_depth_424 has few natural gaps (v2 uses the
sensor on 97.6% / 98.1% of the frames), so gaps are simulated on the recorded clip and
detections (data/results/m5_detections_ego_depth_424.npz; MediaPipe is not run again):
the sensor is withheld (depth_m=None) for every detection inside a span of GAP_S
seconds, the span starting every STEP_S seconds. Scenarios:

  clean        spans anywhere in the clip
  depth_phase  spans entirely inside the recorder's 'depth' phase (hand moved towards and
               away from the camera)
  contaminated as clean, with PnP's wrist distance times BURST_SCALE for the BURST_S
               seconds before the span (PnP placing the hand too close, as on the dry run;
               the ratios go up, the sensor frames themselves do not change)

Variants (all give the same wrist on sensor frames, v2's palm plane):

  v2              WristDepthV2: PnP x median of the last 30 in-band ratios
  v3              vision/depth_wrist_v3.WristDepthV3: last sensor range on the current
                  landmark-0 pixel ray
  v3_pnprel       v3's range times PnP's distance now / at the last sensor frame
                  (= PnP x the last ratio, but on the landmark-0 ray)
  v3_pnprel_med   as v3_pnprel with PnP's distance as the median of its last MED_FRAMES
                  frames and half of its relative change (the most favourable PnP-relative
                  form found while developing, tuned on this clip)
  v3_max_anchor   v3 for MAX_HOLD_S, then PnP times the ratio held range / PnP distance at
                  that moment (continuous at the switch)
  v3_max_recent   v3 for MAX_HOLD_S, then PnP times the median of the last RECENT ratios

Reference: v2 on the same frames without the gap (the sensor wrist). Per gap: the
position error on the gap's frames whose reference is a sensor frame, and the step at the
gap's end (first frame back on the sensor minus the last gap frame). The same through
RayKalmanV2 (vision/depth_filter_v2.py) with sigma_ray 5 mm on "depth" frames and 45 mm
otherwise, as hand_tracker_ego_v3 does live: error over the gap and END_S after it
against the filtered reference, and the largest frame step in the END_S after the gap.
Also the natural hold -> depth steps in the unmasked clip.

The sensor plane is computed once per detection from depth.npy (what WristDepthV3.measure
returns) and replayed; the fusion classes themselves run unchanged.

Usage (with .venv-vision or .venv-arm, from the repository root):
    .venv-vision/bin/python -B tools/eval_depth_wrist_v3.py [ego_depth_424]
Writes data/results/m7_hold_v3.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision.depth_filter import DEFAULT_SIGMA_RAY  # noqa: E402
from vision.depth_filter_v2 import RayKalmanV2  # noqa: E402
from vision.depth_wrist import HOLD_WINDOW, SCALE_BAND  # noqa: E402
from vision.depth_wrist_v2 import WristDepthV2  # noqa: E402
from vision.depth_wrist_v3 import WristDepthV3  # noqa: E402

OUT = REPO / "data" / "results" / "m7_hold_v3.json"
SIDES = ("left", "right")
GAP_S = (0.3, 1.0, 2.0)
#: A span starts every STEP_S seconds, from WARM_S into the clip.
STEP_S = 0.2
WARM_S = 3.0
#: Each run starts WARM_S before the span (v2's 30-ratio window is 1 s) and ends END_S after it.
END_S = 0.5
BURST_SCALE, BURST_S = 0.65, 0.6
MED_FRAMES, MED_GAIN = 15, 0.5
MAX_HOLD_S = 1.0
RECENT = 5
#: hand_tracker_ego_v3.SIGMA_RAY_SENSOR, written out so this also runs without OpenCV.
SIGMA_RAY_SENSOR = 0.005
#: Frames closer than this are consecutive (the recorder runs at ~30 fps).
MAX_STEP_S = 0.05
VARIANTS = ("v2", "v3", "v3_pnprel", "v3_pnprel_med", "v3_max_anchor", "v3_max_recent")


def r(x, n: int = 1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def stats(x, n: int = 1) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {"p50": r(np.percentile(x, 50), n), "p95": r(np.percentile(x, 95), n), "max": r(x.max(), n), "n": int(len(x))}


class Frame:
    """Which detection the cached measure() below should hand back."""

    def __init__(self, cache: list) -> None:
        self.cache, self.j = cache, 0


class CachedV2(WristDepthV2):
    def __init__(self, frame: Frame) -> None:
        super().__init__()
        self.frame = frame

    def measure(self, image_landmarks, depth_m, camera_matrix):
        return self.frame.cache[self.frame.j][0]


class CachedV3(WristDepthV3):
    def __init__(self, frame: Frame) -> None:
        super().__init__()
        self.frame = frame

    def measure(self, image_landmarks, depth_m, camera_matrix):
        return self.frame.cache[self.frame.j]


class OtherHold:
    """WristDepthV3 with its "hold" replaced by one of the PnP-based alternatives above."""

    def __init__(self, frame: Frame, mode: str, fps: float) -> None:
        self.v3, self.mode = CachedV3(frame), mode
        self.max_hold = int(round(MAX_HOLD_S * fps))
        self.pnp_hist: deque[float] = deque(maxlen=MED_FRAMES)
        self.ratios: deque[float] = deque(maxlen=RECENT)
        self.then = self.held = self.anchor = None

    def __call__(self, pose, image, world, depth_m, k, side):
        d = float(np.linalg.norm(pose.position))
        self.pnp_hist.append(d)
        f = self.v3(pose, image, world, depth_m, k, side)
        if f.source == "depth":
            self.ratios.append(f.scale)
            self.then = d if self.mode == "v3_pnprel" else float(np.median(self.pnp_hist))
            self.held, self.anchor = 0, None
            return f
        if f.source != "hold":
            return f
        self.held += 1
        if self.mode == "v3_pnprel":
            q = f.position * (d / self.then)
        elif self.mode == "v3_pnprel_med":
            q = f.position * (float(np.median(self.pnp_hist)) / self.then) ** MED_GAIN
        elif self.held <= self.max_hold:
            q = f.position
        elif self.mode == "v3_max_anchor":
            if self.anchor is None:
                self.anchor = float(np.linalg.norm(f.position)) / d
            q = np.asarray(pose.position) * self.anchor
        else:  # v3_max_recent
            q = np.asarray(pose.position) * float(np.median(self.ratios))
        return dataclasses.replace(f, position=q, scale=float(np.linalg.norm(q)) / d)


def make(variant: str, frame: Frame, fps: float):
    if variant == "v2":
        return CachedV2(frame)
    if variant == "v3":
        return CachedV3(frame)
    return OtherHold(frame, variant, fps)


def run(variant: str, d: dict, frames: np.ndarray, drop: np.ndarray, pnp_scale: np.ndarray, fps: float):
    """One fusion over the detections `frames` (indices into d): positions, sources, filtered positions."""
    frame = Frame(d["cache"])
    fusion, kf = make(variant, frame, fps), RayKalmanV2()
    n = len(frames)
    pos, filt, src = np.empty((n, 3)), np.empty((n, 3)), np.empty(n, dtype=object)
    for a, j in enumerate(frames):
        frame.j = j
        pose = SimpleNamespace(position=d["p"][j] * pnp_scale[a], rotation=d["R"][j])
        f = fusion(pose, d["image"][j], d["world"][j], None if drop[a] else d["dummy"], d["k"], d["side"])
        pos[a], src[a] = f.position, f.source
        filt[a] = kf(f.position, d["t"][j], SIGMA_RAY_SENSOR if f.source == "depth" else DEFAULT_SIGMA_RAY)
    return pos, src, filt


def gaps(d: dict, gap_s: float, scenario: str, fps: float) -> dict:
    t, phase = d["t"], d["phase"]
    res = {v: {"err": [], "jump": [], "err_rk": [], "jump_rk": []} for v in VARIANTS}
    ref_jump, ref_jump_rk, v2_held = [], [], []
    for start in np.arange(t[0] + WARM_S, t[-1] - gap_s - END_S, STEP_S):
        frames = np.flatnonzero((t >= start - WARM_S) & (t < start + gap_s + END_S))
        tw = t[frames]
        gap = (tw >= start) & (tw < start + gap_s)
        if gap.sum() < 3:
            continue
        if scenario == "depth_phase" and not np.all(phase[frames][gap] == "depth"):
            continue
        g0, e = np.flatnonzero(gap)[0], np.flatnonzero(gap)[-1] + 1
        if e >= len(frames) or tw[e] - tw[e - 1] > MAX_STEP_S or tw[g0] - tw[g0 - 1] > MAX_STEP_S:
            continue
        one, nodrop = np.ones(len(frames)), np.zeros(len(frames), bool)
        ref, ref_src, ref_rk = run("v2", d, frames, nodrop, one, fps)
        # the span must end on a sensor frame and follow one, or it is a natural gap, not ours
        if ref_src[e] != "depth" or ref_src[g0 - 1] != "depth":
            continue
        scale = one.copy()
        if scenario == "contaminated":
            scale[(tw >= start - BURST_S) & (tw < start)] = BURST_SCALE
        after = (tw >= tw[e]) & (tw < tw[e] + END_S)
        sensor_gap = gap & (ref_src == "depth")
        ref_jump.append(np.linalg.norm(ref[e] - ref[e - 1]) * 1000)
        ref_jump_rk.append(np.max(np.linalg.norm(np.diff(ref_rk[e - 1:][: after.sum() + 1], axis=0), axis=1)) * 1000)
        for v in VARIANTS:
            pos, src, rk = run(v, d, frames, gap, scale, fps)
            m = res[v]
            m["err"].extend(np.linalg.norm(pos[sensor_gap] - ref[sensor_gap], axis=1) * 1000)
            m["jump"].append(np.linalg.norm(pos[e] - pos[e - 1]) * 1000)
            m["err_rk"].extend(np.linalg.norm(rk[gap | after] - ref_rk[gap | after], axis=1) * 1000)
            m["jump_rk"].append(np.max(np.linalg.norm(np.diff(rk[e - 1:][: after.sum() + 1], axis=0), axis=1)) * 1000)
            if v == "v2" and src[g0] == "hold":
                # the ratio v2 holds, over the true one (reference sensor distance / clean PnP distance)
                true = np.linalg.norm(ref[g0 - 1]) / np.linalg.norm(d["p"][frames[g0 - 1]])
                v2_held.append(np.linalg.norm(pos[g0]) / np.linalg.norm(d["p"][frames[g0]] * scale[g0]) / true)
    out = {"n_gaps": len(ref_jump), "reference_end_step_mm": stats(ref_jump),
           "reference_rk_max_step_after_mm": stats(ref_jump_rk),
           "v2_held_ratio_over_true": stats(v2_held, 3)}
    out["variants"] = {v: {"gap_error_mm": stats(m["err"]), "end_step_mm": stats(m["jump"]),
                           "rk_error_mm": stats(m["err_rk"]), "rk_max_step_after_mm": stats(m["jump_rk"])}
                       for v, m in res.items()}
    return out


def natural(d: dict, fps: float) -> dict:
    """hold -> depth steps in the unmasked clip, per variant (no reference: the sensor had nothing there)."""
    frames = np.arange(len(d["t"]))
    one, nodrop = np.ones(len(frames)), np.zeros(len(frames), bool)
    out = {}
    for v in ("v2", "v3"):
        pos, src, _ = run(v, d, frames, nodrop, one, fps)
        back = np.flatnonzero((src[1:] == "depth") & (src[:-1] == "hold") & (np.diff(d["t"]) < MAX_STEP_S)) + 1
        out[v] = {"by_source": {k: int(c) for k, c in zip(*np.unique(src.astype(str), return_counts=True))},
                  "hold_to_depth_step_mm": stats(np.linalg.norm(pos[back] - pos[back - 1], axis=1) * 1000)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?", default="ego_depth_424", help="clip name under data/clips/")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    clip = REPO / "data" / "clips" / args.clip
    det = np.load(REPO / "data" / "results" / f"m5_detections_{args.clip}.npz")
    meta = json.loads((clip / "meta.json").read_text())
    depth = np.load(clip / "depth.npy", mmap_mode="r")
    if len(depth) != len(det["t"]):
        sys.exit(f"{args.clip}: depth.npy has {len(depth)} rows, detections {len(det['t'])}")
    c = meta["color"]
    k = np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])
    fps = 1.0 / float(np.median(np.diff(det["t"])))
    dummy = np.zeros(depth.shape[1:])  # stands in for the depth image: only its size is read

    sides = {}
    for side in SIDES:
        idx = np.flatnonzero(det[f"{side}_found"] & np.isfinite(det[f"{side}_p"]).all(axis=1))
        image = det[f"{side}_image"][idx]
        measure = WristDepthV3().measure
        cache = [measure(image[j], depth[i].astype(np.float64) * meta["depth_scale"], k) for j, i in enumerate(idx)]
        d = {"side": side, "t": det["t"][idx], "phase": det["phase"][idx], "p": det[f"{side}_p"][idx],
             "R": det[f"{side}_R"][idx], "image": image, "world": det[f"{side}_world"][idx],
             "cache": cache, "k": k, "dummy": dummy}
        dist = [np.linalg.norm(hand[0]) for hand, _ in cache if hand is not None]
        res = {"n_found": int(len(idx)),
               "sensor_wrist_distance_m": {f"p{q}": r(np.percentile(dist, q), 3) for q in (5, 50, 95)},
               "natural": natural(d, fps), "gaps": {}}
        for scenario in ("clean", "depth_phase", "contaminated"):
            for g in GAP_S:
                key = f"{scenario}_{g:g}s"
                res["gaps"][key] = x = gaps(d, g, scenario, fps)
                print(f"{side:5s} {key:18s} gaps {x['n_gaps']:3d}  ref end step {x['reference_end_step_mm']}  "
                      f"v2 held/true {x['v2_held_ratio_over_true']}")
                for v in VARIANTS:
                    m = x["variants"][v]
                    print(f"   {v:14s} err {m['gap_error_mm']}  end {m['end_step_mm']}  "
                          f"rk err {m['rk_error_mm']}  rk step {m['rk_max_step_after_mm']}")
        print(f"{side:5s} natural {res['natural']}")
        sides[side] = res

    payload = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "tool": "tools/eval_depth_wrist_v3.py",
            "clip": f"data/clips/{args.clip}",
            "detections": f"data/results/m5_detections_{args.clip}.npz (tools/extract_ego_detections.py)",
            "intrinsics": "meta.json colour intrinsics (as tools/eval_depth_wrist_v2.py)",
            "reference": "v2 (WristDepthV2) on the same frames without the gap; errors only on gap frames where it used the sensor",
            "gap": {"spans_s": list(GAP_S), "step_s": STEP_S, "warm_s": WARM_S, "after_s": END_S,
                    "rule": "every detection in [start, start + span) gets depth_m=None; the span must follow and "
                            "end on a reference sensor frame, consecutive (< 0.05 s)"},
            "contamination": {"pnp_distance_scale": BURST_SCALE, "seconds_before_gap": BURST_S},
            "variants": {"v2": f"PnP x median of the last {HOLD_WINDOW} in-band ratios",
                         "v3": "last sensor range on the current landmark-0 pixel ray",
                         "v3_pnprel": "v3 range x PnP distance now / at the last sensor frame",
                         "v3_pnprel_med": f"v3 range x (median of last {MED_FRAMES} PnP distances now / at the last "
                                          f"sensor frame) ** {MED_GAIN}",
                         "v3_max_anchor": f"v3 for {MAX_HOLD_S} s, then PnP x (held range / PnP distance at the switch)",
                         "v3_max_recent": f"v3 for {MAX_HOLD_S} s, then PnP x median of the last {RECENT} ratios"},
            "abs_band": list(SCALE_BAND),
            "end_step": "|wrist at the first frame after the gap - wrist at the last gap frame|, mm (natural motion included: see reference_end_step_mm)",
            "ray_kalman": {"filter": "vision/depth_filter_v2.RayKalmanV2 defaults", "sigma_ray_depth_m": SIGMA_RAY_SENSOR,
                           "sigma_ray_other_m": DEFAULT_SIGMA_RAY,
                           "rk_error": "filtered vs filtered reference over the gap and the 0.5 s after",
                           "rk_max_step_after": "largest frame step of the filtered wrist from the last gap frame to 0.5 s after"},
            "v2_held_ratio_over_true": "v2's ratio on the first gap frame / reference sensor distance over clean PnP distance on the frame before",
            "fps": round(fps, 2),
        },
        "sides": sides,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
