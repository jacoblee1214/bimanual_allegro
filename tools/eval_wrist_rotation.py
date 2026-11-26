#!/usr/bin/env python3
"""
eval_wrist_rotation.py — compare ways of getting the palm's orientation, on a recorded clip.

For docs/13_wrist_rotation.md. Input: a tools/record_ego_turns.py clip and its detections
(tools/extract_ego_detections.py). Per frame and hand, the palm frame (vision.wrist_pose.palm_frame:
z along the hand, x out of the palm) from:

  pnp      SQPnP of the 21 landmarks (what the tracker publishes on frames without depth)
  sensor   the depth sensor's palm plane (WristDepthV3; what it publishes on "depth" frames)
  world    MediaPipe's world landmarks as they are, turned from the image centre's viewing
           direction to the wrist's (they are given for a hand seen straight ahead)
  hybrid   the finger direction (z) from `sensor`, the roll about it from `world`;
           `world` alone on frames without a sensor plane

Per phase: how many frames have a value, the step between consecutive frames, the steps over
20 degrees, and the roll about the fingers (range, and how much of the time the palm faces
away from the camera), which in the flip phase should sweep half a turn three times.

Usage (with .venv-vision):
    tools/eval_wrist_rotation.py data/clips/ego_turns
Writes data/results/wrist_rotation_<clip>.json and .npz (the frames per method).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision.depth_wrist_v3 import WristDepthV3  # noqa: E402
from vision.wrist_pose import WristPose, palm_frame  # noqa: E402

METHODS = ("pnp", "sensor", "world", "hybrid")


def toward(ray: np.ndarray) -> np.ndarray:
    """The smallest rotation that takes the optical axis (0, 0, 1) to `ray`."""
    ray = ray / np.linalg.norm(ray)
    axis = np.cross([0.0, 0.0, 1.0], ray)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(axis / s * np.arctan2(s, ray[2])).as_matrix()


def with_roll_from(z: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    """A frame with the given z, and x as close to x_hint as a unit vector across z can be."""
    z = z / np.linalg.norm(z)
    x = x_hint - z * (x_hint @ z)
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def frames(clip: Path, det, side: str) -> dict[str, np.ndarray]:
    meta = json.loads((clip / "meta.json").read_text())
    c = meta["color"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
    depth = np.load(clip / "depth.npy", mmap_mode="r")
    n = len(det["t"])
    out = {m: np.full((n, 3, 3), np.nan) for m in METHODS}
    source = np.array([""] * n, dtype=object)
    fusion = WristDepthV3()
    for i in range(n):
        if not det[f"{side}_found"][i] or np.isnan(det[f"{side}_p"][i]).any():
            fusion.reset()
            continue
        image, world, p = det[f"{side}_image"][i], det[f"{side}_world"][i], det[f"{side}_p"][i]
        pose = WristPose(p, det[f"{side}_R"][i], float(det[f"{side}_reproj"][i]))
        fused = fusion(pose, image, world, depth[i].astype(np.float64) * meta["depth_scale"], K, side)
        source[i] = fused.source
        out["pnp"][i] = pose.rotation
        ray = np.linalg.solve(K, [image[0, 0] * c["width"], image[0, 1] * c["height"], 1.0])
        out["world"][i] = toward(ray) @ palm_frame(world, side)
        if fused.rotation is not None:
            out["sensor"][i] = fused.rotation
            out["hybrid"][i] = with_roll_from(fused.rotation[:, 2], out["world"][i][:, 0])
        else:
            out["hybrid"][i] = out["world"][i]
    out["source"] = source
    return out


def summarize(R: np.ndarray, t: np.ndarray, keep: np.ndarray) -> dict:
    have = keep & ~np.isnan(R[:, 0, 0])
    idx = np.where(have)[0]
    if len(idx) < 3:
        return {"frames": int(have.sum()), "of": int(keep.sum())}
    pairs = (np.diff(idx) == 1) & (np.diff(t[idx]) < 0.05)
    a, b = Rotation.from_matrix(R[idx[:-1]][pairs]), Rotation.from_matrix(R[idx[1:]][pairs])
    step = np.degrees((a.inv() * b).magnitude())
    x, z = R[idx][:, :, 0], R[idx][:, :, 2]
    # the step split into the turn of the finger direction and the roll about it
    finger = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", R[idx[:-1]][pairs][:, :, 2], R[idx[1:]][pairs][:, :, 2]), -1, 1)))
    return {"frames": int(have.sum()), "of": int(keep.sum()),
            "step_deg_p50_p95_max": np.percentile(step, [50, 95, 100]).round(1).tolist(),
            "steps_over_20": int((step > 20).sum()),
            "finger_step_deg_p95": round(float(np.percentile(finger, 95)), 1),
            "palm_away_from_camera_pct": round(100 * float(np.mean(x[:, 2] > 0)), 0),
            "finger_dir_mean": z.mean(0).round(2).tolist()}


def main() -> None:
    clip = Path(sys.argv[1])
    det = np.load(REPO / "data" / "results" / f"m5_detections_{clip.name}.npz", allow_pickle=True)
    t, phase = det["t"], det["phase"]
    result, saved = {}, {"t": t, "phase": phase}
    for side in ("left", "right"):
        f = frames(clip, det, side)
        src = f.pop("source")
        saved.update({f"{side}_{m}": f[m] for m in METHODS})
        saved[f"{side}_source"] = src.astype(str)
        for ph in dict.fromkeys(phase):
            keep = phase == ph
            found = det[f"{side}_found"][keep]
            depth_frames = int((src[keep] == "depth").sum())
            print(f"== {side} / {ph}: hand found {int(found.sum())}/{int(keep.sum())}, sensor plane on {depth_frames}")
            for m in METHODS:
                s = summarize(f[m], t, keep)
                result[f"{side}/{ph}/{m}"] = s
                if "step_deg_p50_p95_max" in s:
                    print(f"   {m:7s} frames {s['frames']:4d}  step p50/p95/max {s['step_deg_p50_p95_max']}  over 20 deg: {s['steps_over_20']:3d}"
                          f"  finger-direction step p95 {s['finger_step_deg_p95']:5.1f}  palm away from camera {s['palm_away_from_camera_pct']:.0f}%")
    out = REPO / "data" / "results" / f"wrist_rotation_{clip.name}"
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), **saved)
    print(f"-> {out}.json, .npz")


if __name__ == "__main__":
    main()
