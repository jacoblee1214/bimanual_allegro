#!/usr/bin/env python3
"""
eval_node_v4.py — milestone M6: vision/hand_tracker_ego_v3.py (what hand_tracker_node_v4.py
publishes with --depth --fusion v1) over a tools/record_ego_depth.py clip, without ROS.

Two runs:

  replay  M5's recorded detections (data/results/m5_detections_<clip>.npz, solved with
          config/ego_camera_info.yaml) fed to the tracker through a stand-in detector, with
          the clip's depth. Its published wrist position is compared frame by frame with
          M5's offline `fused_rk` (tools/eval_depth_wrist.py: DepthFusion with the clip's
          meta.json intrinsics, then RayKalman with 5 mm / 45 mm ray noise), recomputed
          here from the same detections, and with the still-phase numbers stored in
          data/results/m5_depth_wrist.json.
  live    MediaPipe on the GPU over color.avi, intrinsics from meta.json (what the node
          uses for clip:<dir>), as fast as it goes. Still-phase jitter and lag as M5
          measures them, for the tracker's output (`fused_rk`) and for the same detections
          without depth (`raw_rk`, which is hand_tracker_node_v3 --ray-filter), and the
          time per frame: detection, the rest of process() (PnP + fusion + filters), and
          the depth conversion to metres.

Writes data/results/m6_node_v4.json.

Usage (with .venv-vision, from the repository root):
    .venv-vision/bin/python -B tools/eval_node_v4.py [ego_depth_424]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import tools.eval_depth_wrist as m5  # noqa: E402
from vision import camera  # noqa: E402
from vision.depth_filter import DEFAULT_SIGMA_RAY  # noqa: E402
from vision.depth_wrist import DepthFusion, palm_depth  # noqa: E402
from vision.hand_tracker import SIDES, RawHand  # noqa: E402
from vision.hand_tracker_ego_v3 import SIGMA_RAY_SENSOR, DepthEgoHandTracker  # noqa: E402
from vision.rgbd_source import ClipRGBD  # noqa: E402

OUT = REPO / "data" / "results" / "m6_node_v4.json"
#: Frames left out of the timing while the GPU delegate warms up.
WARMUP = 10


class Recorded:
    """Stand-in detector: hands back one recorded list of RawHand per call."""

    def __init__(self, frames: list[list[RawHand]]) -> None:
        self.frames, self.i = frames, 0

    def __call__(self, rgb, t):
        self.i += 1
        return self.frames[self.i - 1]

    def close(self) -> None:
        pass


class Timed:
    """Wraps the tracker's detector: times it and keeps what it returned."""

    def __init__(self, detect) -> None:
        self.detect, self.ms, self.frames = detect, [], []

    def __call__(self, rgb, t):
        t0 = time.perf_counter()
        out = self.detect(rgb, t)
        self.ms.append((time.perf_counter() - t0) * 1000.0)
        self.frames.append(out)
        return out

    def close(self) -> None:
        self.detect.close()


def run(tracker: DepthEgoHandTracker, src: ClipRGBD, use_depth: bool = True) -> dict:
    """Per side: found mask, published position, unfiltered fused and raw PnP position, source; and ms per frame."""
    n = len(src.rows)
    out = {s: {"found": np.zeros(n, bool), "p": np.full((n, 3), np.nan), "fused": np.full((n, 3), np.nan),
               "raw": np.full((n, 3), np.nan), "source": np.full(n, "", dtype=object)} for s in SIDES}
    ms = []
    src.rewind()
    for i in range(n):
        ok, frame, depth = src.read()
        if not ok:
            sys.exit(f"{src.clip}: ends at frame {i}")
        t0 = time.perf_counter()
        hands = tracker.process(frame, src.t, depth if use_depth else None)
        ms.append((time.perf_counter() - t0) * 1000.0)
        for side, h in hands.items():
            if h.wrist is None:
                continue
            o = out[side]
            o["found"][i], o["p"][i], o["fused"][i] = True, h.wrist.position, h.fused.position
            o["raw"][i], o["source"][i] = h.fused.position / h.fused.scale, h.fused.source
    return {"sides": out, "ms": np.array(ms)}


def metrics(p: np.ndarray, unfiltered: np.ndarray, raw: np.ndarray, t: np.ndarray, phase: np.ndarray) -> dict:
    """M5's still-phase jitter and lag (tools/eval_depth_wrist.py) for one series over its found frames."""
    r_med = float(np.median(np.linalg.norm(raw, axis=1)))
    bearing = lambda q: q / np.linalg.norm(q, axis=1, keepdims=True) * r_med  # noqa: E731
    return {
        "still": m5.still_jitter(p, t, phase == "still"),
        "lag_lateral_vs_raw_ms": m5.lag_ms(t, bearing(raw), bearing(p)),
        "lag_ray_vs_unfiltered_ms": m5.lag_ms(t, np.linalg.norm(unfiltered, axis=1), np.linalg.norm(p, axis=1)),
    }


def ms_stats(x: np.ndarray) -> dict:
    x = np.asarray(x)[WARMUP:]
    return {"p50": m5.r(np.percentile(x, 50)), "p95": m5.r(np.percentile(x, 95)), "max": m5.r(x.max()), "n": int(len(x))}


def replay(name: str, clip: Path, meta: dict, k_meta: np.ndarray) -> dict:
    det = np.load(REPO / "data" / "results" / f"m5_detections_{name}.npz")
    t, phase = det["t"], det["phase"]
    found = {s: det[f"{s}_found"] & np.isfinite(det[f"{s}_p"]).all(axis=1) for s in SIDES}
    frames = [[RawHand(det[f"{s}_image"][i], det[f"{s}_world"][i], "", float(det[f"{s}_score"][i]))
               for s in SIDES if found[s][i]] for i in range(len(t))]
    cal_pnp = camera.load(REPO / "config" / "ego_camera_info.yaml")  # what the detections were solved with
    src = ClipRGBD(clip)
    res = run(DepthEgoHandTracker(cal_pnp, detector=Recorded(frames)), src)
    depth = np.load(clip / "depth.npy", mmap_mode="r")
    stored = json.loads((REPO / "data" / "results" / "m5_depth_wrist.json").read_text())["clips"].get(name, {})

    out = {}
    for s in SIDES:
        o = res["sides"][s]
        idx = np.flatnonzero(found[s])
        # M5's fused_rk, recomputed from the detections exactly as tools/eval_depth_wrist.py does
        p_raw, rot, world, image = det[f"{s}_p"][idx], det[f"{s}_R"][idx], det[f"{s}_world"][idx], det[f"{s}_image"][idx]
        fusion, p_fused, used = DepthFusion(), np.empty_like(p_raw), np.zeros(len(idx), bool)
        for j, i in enumerate(idx):
            f = fusion(p_raw[j], rot[j], world[j], palm_depth(depth[i].astype(np.float64) * meta["depth_scale"], image[j]),
                       k_meta, s)
            p_fused[j], used[j] = f.position, f.source == "depth"
        ref = m5.run_kalman(p_fused, t[idx], np.where(used, SIGMA_RAY_SENSOR, DEFAULT_SIGMA_RAY))

        same_frames = bool(np.array_equal(o["found"], found[s]))
        mine = o["p"][idx]
        diff = np.linalg.norm(mine - ref, axis=1) * 1000
        differ_source = np.flatnonzero((o["source"][idx] == "depth") != used)
        out[s] = {
            "n_found": int(len(idx)),
            "same_found_frames": same_frames,
            "position_diff_vs_m5_fused_rk_mm": {"max": m5.r(diff.max(), 4), "p99": m5.r(np.percentile(diff, 99), 4),
                                                "p50": m5.r(np.percentile(diff, 50), 4)},
            "frames_whose_sensor_use_differs": int(len(differ_source)),
            "first_frame_whose_sensor_use_differs_t": m5.r(t[idx][differ_source[0]], 4) if len(differ_source) else None,
            "fused_from_sensor": m5.r((o["source"][idx] == "depth").mean(), 4),
            "tracker": metrics(mine, o["fused"][idx], o["raw"][idx], t[idx], phase[idx]),
            "m5_recomputed": metrics(ref, p_fused, p_raw, t[idx], phase[idx]),
            "m5_stored_fused_rk": {k: stored.get("sides", {}).get(s, {}).get("variants", {}).get("fused_rk", {}).get(k)
                                   for k in ("still", "lag_lateral_vs_raw_ms", "lag_ray_vs_unfiltered_ms")},
        }
    src.release()
    return out


def live(clip: Path) -> tuple[dict, dict]:
    src = ClipRGBD(clip)
    tracker = DepthEgoHandTracker(src.calibration, gpu=True)
    timed = Timed(tracker._detect)
    tracker._detect = timed
    fused = run(tracker, src)
    tracker.close()
    plain = run(DepthEgoHandTracker(src.calibration, detector=Recorded(timed.frames)), src, use_depth=False)

    t = src.times
    phase = np.array([r["phase"] for r in src.rows])
    out = {}
    for s in SIDES:
        f, p = fused["sides"][s], plain["sides"][s]
        m = f["found"]
        out[s] = {
            "n_found": int(m.sum()),
            "same_found_frames_without_depth": bool(np.array_equal(m, p["found"])),
            "fused_from_sensor": m5.r((f["source"][m] == "depth").mean(), 4),
            "by_source": {k: int(v) for k, v in zip(*np.unique(f["source"][m], return_counts=True))},
            "fused_rk": metrics(f["p"][m], f["fused"][m], f["raw"][m], t[m], phase[m]),
            "raw_rk": metrics(p["p"][m], p["raw"][m], p["raw"][m], t[m], phase[m]),
        }

    # the conversion the clip and the RealSense both do, on frames already in memory
    depth = np.load(clip / "depth.npy", mmap_mode="r")
    conv = []
    for i in range(0, len(depth), 4):
        d = np.array(depth[i])
        t0 = time.perf_counter()
        _ = d.astype(np.float32) * np.float32(src.depth_scale)
        conv.append((time.perf_counter() - t0) * 1000.0)
    src.release()
    timing = {
        "process_total_ms": ms_stats(fused["ms"]),
        "detect_ms": ms_stats(timed.ms),
        "process_minus_detect_ms": ms_stats(fused["ms"] - np.array(timed.ms)),
        "process_recorded_detections_no_depth_ms": ms_stats(plain["ms"]),
        "depth_to_metres_ms": {"p50": m5.r(np.percentile(conv, 50), 3), "p95": m5.r(np.percentile(conv, 95), 3), "n": len(conv)},
        "note": f"first {WARMUP} frames left out; run as fast as possible, not paced; "
                "process_recorded_detections_no_depth_ms: the same tracker on the detections just recorded (detector cost about 0), no depth: colour conversion + PnP + filters",
    }
    return out, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", nargs="?", default="ego_depth_424", help="clip name under data/clips/")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    clip = REPO / "data" / "clips" / args.clip
    meta = json.loads((clip / "meta.json").read_text())
    c = meta["color"]
    k_meta = np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])

    rep = replay(args.clip, clip, meta, k_meta)
    for s in SIDES:
        x = rep[s]
        print(f"replay {s:5s} same frames {x['same_found_frames']}  diff vs M5 fused_rk mm {x['position_diff_vs_m5_fused_rk_mm']}  "
              f"sensor-use differs {x['frames_whose_sensor_use_differs']}")
        print(f"    tracker still {x['tracker']['still']}")
        print(f"    m5      still {x['m5_stored_fused_rk']['still']}")
    liv, timing = live(clip)
    for s in SIDES:
        x = liv[s]
        print(f"live   {s:5s} found {x['n_found']} sensor {x['fused_from_sensor']}  fused_rk {x['fused_rk']}")
        print(f"                raw_rk {x['raw_rk']}")
    print(json.dumps(timing, indent=1))

    payload = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "tool": "tools/eval_node_v4.py",
            "clip": f"data/clips/{args.clip}",
            "tracker": "vision/hand_tracker_ego_v3.DepthEgoHandTracker, fusion DepthFusionV1 (M5 palm_depth + DepthFusion)",
            "sigma_ray_m": {"sensor_frames": SIGMA_RAY_SENSOR, "other_frames": DEFAULT_SIGMA_RAY},
            "replay": "M5 recorded detections, PnP intrinsics config/ego_camera_info.yaml (as M5); the tracker uses them "
                      "for the fusion too, M5 used meta.json's (fx differs by about 0.1 px)",
            "live": "MediaPipe Tasks API on the GPU over color.avi, intrinsics from meta.json (as the node for clip:<dir>)",
            "still": "frames whose recorder phase is 'still' (tools/eval_depth_wrist.still_jitter)",
        },
        "replay": rep,
        "live": liv,
        "timing": timing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
