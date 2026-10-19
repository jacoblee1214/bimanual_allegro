#!/usr/bin/env python3
"""
analyze_scale_jitter.py — experiment A of docs/04_rnd_candidates_survey.md (5.1, steps 1-3):
does frame-to-frame wobble of MediaPipe's hand size make the wrist jitter in depth?

The wrist position is PnP between MediaPipe's metric world landmarks and the
image landmarks (vision/wrist_pose.py), so its distance from the camera scales
with the size of the world-landmark hand. Per clip and side, from the raw
per-frame detection (HandTracker.detect and the tracker's side assignment, no
One Euro filter), this records

  * the palm scale s: mean distance from the wrist (0) to the MCPs 5, 9, 13, 17
    in the world landmarks (metacarpals, which barely change as the hand closes);
  * the raw PnP wrist position p,

and then

  1. splits the wrist jitter, on still stretches, into the component along the
     camera ray to the wrist (depth) and the rest (lateral);
  2. correlates relative scale change with relative depth change (if the scale
     wobble is the cause the slope is about +1: a bigger hand is placed farther);
  3. rescales the world landmarks to a reference scale before PnP and measures
     again: the session (clip, side) median, which is offline only, and causal
     references usable live (median over the past N detections, and over all
     past detections of that side);
  4. runs the tracker's position One Euro filter (current defaults) over the raw
     and the rescaled positions, for comparison.

Still: a detection is still when it and the STILL_HALF detections either side
are consecutive frames, and the palm's image centroid (landmarks 0, 5, 9, 13,
17, pixels) moves less than --still-px between each pair of them. Only image
motion across the screen is used, not the hand's apparent size, so the
criterion does not pick frames for low depth noise.

Jitter metrics, per position series, over still detections:
  * d2: second difference p[j+1] - 2 p[j] + p[j-1] (the V5 definition, which
    bench_tracker.py takes over all frames); its component along u = p[j]/|p[j]|
    is depth, the rest lateral; median of the absolute values, mm;
  * dev: p[j] minus the 5-detection centred moving median; same split; RMS, mm.
Plus `v5_style`: bench_tracker.py's own figure (median over runs of the median
|d2|, all frames, runs over 10 detections), so the V5 numbers can be checked.

Intrinsics are the nominal guess tools/extract_wrist_trajectories.py uses
(fx = fy = --fx, centre of the image, no distortion), so mm are relative.
Timestamps come from the clip's .jsonl (the real capture times, ~18 fps), which
the One Euro filter needs; without it, frame / video fps.

Usage:
    tools/analyze_scale_jitter.py                  # legacy CPU detector, the four V2 clips
    tools/analyze_scale_jitter.py --gpu            # Tasks API on the GPU
Results are merged into data/results/m2_scale_jitter.json under the detector's key
("cpu", "gpu"; a non-default --still-px adds a suffix, e.g. "cpu_still5px").
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
from vision.filters import OneEuroFilter  # noqa: E402
from vision.hand_tracker import POSITION_FILTER, RESET_GAP_S, SIDES, HandTracker  # noqa: E402
from vision.handedness import Detection, assign_sides  # noqa: E402
from vision.wrist_pose import solve_wrist_pose  # noqa: E402

OUT = REPO / "data" / "results" / "m2_scale_jitter.json"
CLIPS = ["palm2", "back", "cross", "palm"]
MCPS = [5, 9, 13, 17]
PALM = [0, 5, 9, 13, 17]
STILL_HALF = 2          # detections either side that must also be still
MEDIAN_HALF = 2         # moving median over 2 * MEDIAN_HALF + 1 detections
CAUSAL_N = (15, 45)     # past-N-detection medians (~0.8 s and ~2.5 s at 18 fps)


def palm_scale(world: np.ndarray) -> np.ndarray:
    """(n,) mean wrist-to-MCP distance, metres, from (n, 21, 3) world landmarks."""
    return np.linalg.norm(world[:, MCPS] - world[:, [0]], axis=2).mean(axis=1)


def detect_clip(path: Path, gpu: bool) -> tuple[np.ndarray, list[dict]]:
    """Per frame: {side: (world (21, 3), image (21, 3))}, sides as the tracker assigns them."""
    log = path.with_suffix(".jsonl")
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ts = [json.loads(line)["t"] for line in log.open()] if log.exists() else []
    if len(ts) != n:
        ts = [i / fps for i in range(n)]
    tracker = HandTracker(gpu=gpu)
    rows = []
    for t in ts:
        ok, frame = cap.read()
        if not ok:
            break
        raw = tracker.detect(frame, t)
        dets = [Detection(r.label, r.score, r.image_landmarks[0, 0]) for r in raw]
        rows.append({side: (raw[i].world_landmarks, raw[i].image_landmarks)
                     for side, i in assign_sides(dets, label_to_side=tracker.label_to_side).items()})
    tracker.close()
    cap.release()
    return np.asarray(ts[:len(rows)]), rows


def solve_all(world: np.ndarray, image: np.ndarray, k: np.ndarray, size, K, side) -> tuple[np.ndarray, np.ndarray]:
    """Wrist positions (n, 3) and reprojection errors (n,) with the world landmarks scaled by k (n,)."""
    p, e = np.full((len(world), 3), np.nan), np.full(len(world), np.nan)
    for j in range(len(world)):
        pose = solve_wrist_pose(world[j] * k[j], image[j], size, K, side)
        if pose is not None:
            p[j], e[j] = pose.position, pose.reprojection_px
    return p, e


def one_euro(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The tracker's position filter (vision/hand_tracker.py, _PoseFilter) over one side's detections."""
    f, out, last = OneEuroFilter(**POSITION_FILTER), np.empty_like(p), None
    for j in range(len(p)):
        if last is not None and t[j] - last > RESET_GAP_S:
            f.reset()
        out[j], last = f(p[j], t[j]), t[j]
    return out


def moving_median(x: np.ndarray, frames: np.ndarray, half: int) -> np.ndarray:
    """Centred moving median over `half` detections either side; NaN unless all are consecutive frames."""
    out = np.full(x.shape, np.nan)
    for j in range(half, len(x) - half):
        if frames[j + half] - frames[j - half] == 2 * half:
            out[j] = np.median(x[j - half:j + half + 1], axis=0)
    return out


def split(v: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|component along the ray to p| and |the rest|, row-wise."""
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    along = np.sum(v * u, axis=1)
    return np.abs(along), np.linalg.norm(v - along[:, None] * u, axis=1)


def jitter(p: np.ndarray, frames: np.ndarray, mask: np.ndarray) -> dict:
    """d2 and dev metrics (mm) over the detections in `mask` (which must have neighbours, see still_mask)."""
    j = np.flatnonzero(mask)
    d2 = p[j + 1] - 2 * p[j] + p[j - 1]
    d2_depth, d2_lat = split(d2, p[j])
    dev = p - moving_median(p, frames, MEDIAN_HALF)
    dev_depth, dev_lat = split(dev[j], p[j])
    rms = lambda a: float(np.sqrt(np.mean(a ** 2))) * 1000  # noqa: E731
    return {
        "d2_depth_mm": float(np.median(d2_depth)) * 1000, "d2_lateral_mm": float(np.median(d2_lat)) * 1000,
        "d2_total_mm": float(np.median(np.linalg.norm(d2, axis=1))) * 1000,
        "dev_depth_rms_mm": rms(dev_depth), "dev_lateral_rms_mm": rms(dev_lat),
    }


def v5_style(p: np.ndarray, frames: np.ndarray) -> float:
    """bench_tracker.second_diff_mm: runs of consecutive detections, median |d2| per run, median over runs."""
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    per_run = [np.median(np.linalg.norm(np.diff(r, n=2, axis=0), axis=1))
               for r in np.split(p, breaks) if len(r) > 10]
    return float(np.median(per_run)) * 1000 if per_run else float("nan")


def still_mask(image: np.ndarray, frames: np.ndarray, size, still_px: float) -> np.ndarray:
    """Detections whose STILL_HALF neighbours either side are consecutive frames with palm centroid steps < still_px."""
    c = (image[:, PALM, :2] * size).mean(axis=1)
    step = np.full(len(c), np.inf)   # step[j]: motion from detection j-1 to j
    step[1:] = np.where(np.diff(frames) == 1, np.linalg.norm(np.diff(c, axis=0), axis=1), np.inf)
    mask = np.zeros(len(c), dtype=bool)
    for j in range(STILL_HALF, len(c) - STILL_HALF):
        mask[j] = np.all(step[j - STILL_HALF + 1:j + STILL_HALF + 1] < still_px)
    return mask


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return {"n": int(len(x)), "slope": None, "r": None}
    return {"n": int(len(x)), "slope": float(np.polyfit(x, y, 1)[0]), "r": float(np.corrcoef(x, y)[0, 1])}


def analyse_side(t, frames, world, image, size, K, side, still_px) -> dict:
    s = palm_scale(world)
    refs = {"raw": np.ones_like(s), "session": np.full_like(s, np.median(s))}
    for n in CAUSAL_N:
        refs[f"causal{n}"] = np.array([np.median(s[max(0, j - n + 1):j + 1]) for j in range(len(s))])
    refs["causal_all"] = np.array([np.median(s[:j + 1]) for j in range(len(s))])
    for name in refs:
        if name != "raw":
            refs[name] = refs[name] / s

    p_raw, reproj = solve_all(world, image, refs["raw"], size, K, side)
    ok = np.isfinite(p_raw).all(axis=1)
    t, frames, s, image, p_raw, reproj = t[ok], frames[ok], s[ok], image[ok], p_raw[ok], reproj[ok]
    world = world[ok]
    refs = {k: v[ok] for k, v in refs.items()}
    still = still_mask(image, frames, size, still_px)

    variants, check = {}, {}
    for name, k in refs.items():
        p = p_raw if name == "raw" else solve_all(world, image, k, size, K, side)[0]
        if name != "raw":   # PnP is scale-equivariant: scaling the hand by k should move the wrist by exactly k
            check[name] = float(np.nanmax(np.linalg.norm(p - k[:, None] * p_raw, axis=1))) * 1000
        variants[name] = p
    for name in ["raw", "session", "causal15", "causal_all"]:
        variants[f"euro_{name}"] = one_euro(variants[name], t)

    r = np.linalg.norm(p_raw, axis=1)
    cons = np.r_[False, np.diff(frames) == 1]
    dlog_s = np.r_[np.nan, np.diff(np.log(s))]
    dlog_r = np.r_[np.nan, np.diff(np.log(r))]
    dlog_s[~cons] = dlog_r[~cons] = np.nan
    # The hand's size in the image (pixels), same landmarks: for a rigid hand d log r = d log s - d log a,
    # so depth only wobbles with s when a does not move with it.
    a = np.linalg.norm((image[:, MCPS, :2] - image[:, [0], :2]) * size, axis=2).mean(axis=1)
    dlog_a = np.r_[np.nan, np.diff(np.log(a))]
    dlog_a[~cons] = np.nan
    mlog_s = np.log(s) - np.log(moving_median(s, frames, MEDIAN_HALF))
    mlog_r = np.log(r) - np.log(moving_median(r, frames, MEDIAN_HALF))

    return {
        "n_detections": int(len(s)), "n_still": int(still.sum()),
        "scale_median_mm": float(np.median(s)) * 1000,
        "scale_cv": float(np.std(s) / np.mean(s)),
        "scale_step_rel_std_still": float(np.nanstd(dlog_s[still])),
        "depth_step_rel_std_still": float(np.nanstd(dlog_r[still])),
        "image_size_step_rel_std_still": float(np.nanstd(dlog_a[still])),
        "fit_image_size_vs_scale_step_still": fit(dlog_s[still], dlog_a[still]),
        "depth_median_m": float(np.median(r)),
        "reproj_median_px": float(np.median(reproj)),
        "fit_step_still": fit(dlog_s[still], dlog_r[still]),
        "fit_dev_still": fit(mlog_s[still], mlog_r[still]),
        "fit_step_all": fit(dlog_s, dlog_r),
        "jitter_still": {name: jitter(p, frames, still) for name, p in variants.items()},
        "v5_style_mm": {name: v5_style(p, frames) for name, p in variants.items()},
        "pnp_scaling_check_max_mm": check,
        "frames": {"frame": frames.tolist(), "t": t.tolist(), "scale_m": s.tolist(),
                   "p_raw_m": np.round(p_raw, 6).tolist(), "reproj_px": np.round(reproj, 3).tolist(),
                   "still": still.astype(int).tolist()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", default=CLIPS, help="V2 clip names under data/clips")
    p.add_argument("--gpu", action="store_true", help="MediaPipe Tasks API on the NVIDIA GPU (default: legacy CPU)")
    p.add_argument("--fx", type=float, default=600.0, help="nominal focal length in pixels (default 600)")
    p.add_argument("--still-px", type=float, default=3.0, help="still threshold, palm centroid px per frame (default 3)")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    detector = "gpu" if args.gpu else "cpu"
    key = detector if args.still_px == 3.0 else f"{detector}_still{args.still_px:g}px"
    results = {}
    for name in args.clips:
        path = REPO / "data" / "clips" / f"v2_{name}.avi"
        if not path.exists():
            print(f"[!] {path} missing, skipped")
            continue
        ts, rows = detect_clip(path, args.gpu)
        h, w = cv2.VideoCapture(str(path)).read()[1].shape[:2]
        K = np.array([[args.fx, 0, w / 2.0], [0, args.fx, h / 2.0], [0, 0, 1]])
        results[name] = {}
        for side in SIDES:
            idx = np.array([i for i, r in enumerate(rows) if side in r])
            if len(idx) < 2 * STILL_HALF + 2:
                continue
            world = np.array([rows[i][side][0] for i in idx])
            image = np.array([rows[i][side][1] for i in idx])
            res = analyse_side(ts[idx], idx, world, image, (w, h), K, side, args.still_px)
            results[name][side] = res
            js, f = res["jitter_still"], res["fit_step_still"]
            slope = "-" if f["slope"] is None else f"{f['slope']:+.2f} (r {f['r']:+.2f})"
            print(f"{name:6s} {side:5s} n={res['n_detections']:3d} still={res['n_still']:3d} "
                  f"scale cv={res['scale_cv']:.3f} | d2 depth/lat mm raw {js['raw']['d2_depth_mm']:.2f}/"
                  f"{js['raw']['d2_lateral_mm']:.2f} session {js['session']['d2_depth_mm']:.2f}/"
                  f"{js['session']['d2_lateral_mm']:.2f} causal15 {js['causal15']['d2_depth_mm']:.2f} "
                  f"euro {js['euro_raw']['d2_depth_mm']:.2f} | slope {slope}")

    doc = json.loads(args.out.read_text()) if args.out.exists() else {}
    doc[key] = {
        "meta": {
            "date": datetime.date.today().isoformat(), "tool": "tools/analyze_scale_jitter.py",
            "detector": "Tasks API HandLandmarker, GPU" if args.gpu else "legacy solutions.hands, full model, CPU",
            "intrinsics": f"nominal fx=fy={args.fx}, centre of image, no distortion",
            "still": f"palm centroid (0,5,9,13,17) step < {args.still_px} px/frame for the detection and "
                     f"{STILL_HALF} either side, all consecutive frames",
            "moving_median": 2 * MEDIAN_HALF + 1, "causal_n": list(CAUSAL_N),
            "one_euro_position": POSITION_FILTER, "reset_gap_s": RESET_GAP_S,
            "timestamps": "clip .jsonl capture times",
        },
        "clips": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"-> {args.out} [{key}]")


if __name__ == "__main__":
    main()
