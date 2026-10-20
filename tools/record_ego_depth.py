#!/usr/bin/env python3
"""
record_ego_depth.py — record the egocentric RealSense's colour and depth together (USB 3 only).

The depth sensor is the reference that monocular wrist depth (PnP, vision/wrist_pose.py)
has never had: docs/06_wrist_depth_survey.md. This records what is needed to compare
the two offline. Camera only: no ROS, no robot.

A preview window (unmirrored) tells the operator what to do:
   5 s  get ready: both hands in view
   8 s  BOTH hands still            -> still-hand jitter
  10 s  BOTH hands forward and back -> depth tracking
  10 s  BOTH hands free, slowly     -> everything else

Written to data/clips/<name>/ (git-ignored):
  color.avi    MJPG, one frame per row of frames.jsonl
  depth.npy    uint16 (N, 480, 640), aligned to the colour camera; metres = value * depth_scale
  frames.jsonl {"i", "t" (s since start, host clock), "t_color_ms", "t_depth_ms" (device), "phase"}
  meta.json    colour intrinsics, depth_scale, depth preset, R_CB from config/ego_camera.yaml

The depth stream runs at 424x240 by default and is aligned up to the colour frame. The
D457's 95 mm baseline and 126 px disparity search put the closest measurable distance at
fx * 0.095 / 126: about 0.30 m at 640x480 (fx 392), where the backs of hands held near the
body came out empty (clip ego_depth), and about 0.16 m at 424x240 (fx 217).

Usage (with .venv-vision):
    tools/record_ego_depth.py [--name ego_depth] [--depth-res 424x240]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

REPO = Path(__file__).resolve().parents[1]
W, H, FPS = 640, 480, 30
PHASES = (("ready", 5.0, "get ready: BOTH hands in view"), ("still", 8.0, "BOTH hands STILL"),
          ("depth", 10.0, "BOTH hands slowly FORWARD and BACK"), ("free", 10.0, "BOTH hands free, slowly"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="ego_depth")
    p.add_argument("--depth-res", default="424x240", help="depth stream WxH (see above for the closest range)")
    args = p.parse_args()
    dw, dh = (int(v) for v in args.depth_res.split("x"))

    out = REPO / "data" / "clips" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, FPS)
    profile = pipe.start(cfg)
    dev = profile.get_device()
    if not dev.get_info(rs.camera_info.usb_type_descriptor).startswith("3"):
        pipe.stop()
        sys.exit("camera is not on USB 3: colour + depth will not keep up (docs/05_ego_camera.md, 4)")
    depth_sensor = dev.first_depth_sensor()
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_fx = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().fx

    n_max = int(sum(length for _, length, _ in PHASES) * FPS * 1.1)
    depth = np.lib.format.open_memmap(out / "depth.npy", mode="w+", dtype=np.uint16, shape=(n_max, H, W))
    video = cv2.VideoWriter(str(out / "color.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    log = (out / "frames.jsonl").open("w")
    for _ in range(30):  # let auto exposure settle
        pipe.wait_for_frames(3000)
    i, t0 = 0, time.monotonic()
    try:
        for name, length, text in PHASES:
            end = time.monotonic() + length
            while time.monotonic() < end and i < n_max:
                frames = align.process(pipe.wait_for_frames(3000))
                c, d = frames.get_color_frame(), frames.get_depth_frame()
                if not c or not d:
                    continue
                colour = np.asanyarray(c.get_data())
                video.write(colour)
                depth[i] = np.asanyarray(d.get_data())
                log.write(json.dumps({"i": i, "t": round(time.monotonic() - t0, 4), "t_color_ms": c.get_timestamp(),
                                      "t_depth_ms": d.get_timestamp(), "phase": name}) + "\n")
                i += 1
                preview = colour.copy()
                cv2.putText(preview, f"{text}  ({end - time.monotonic():.0f}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("ego depth", preview)
                cv2.waitKey(1)
    finally:
        pipe.stop()
        video.release()
        log.close()
        cv2.destroyAllWindows()
        depth.flush()

    # drop the unused tail of the preallocated array (written beside it, never over the open memmap)
    kept = np.array(depth[:i])
    del depth
    with (out / "depth.npy.part").open("wb") as f:
        np.save(f, kept)
    (out / "depth.npy.part").replace(out / "depth.npy")
    meta = {
        "frames": i,
        "color": {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                  "cx": intr.ppx, "cy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs)},
        "depth_scale": depth_sensor.get_depth_scale(),
        "depth_preset": depth_sensor.get_option_value_description(rs.option.visual_preset,
                                                                 depth_sensor.get_option(rs.option.visual_preset)),
        "depth_aligned_to": "color",
        "depth_stream": {"width": dw, "height": dh, "fx": depth_fx},
        "R_CB": yaml.safe_load((REPO / "config" / "ego_camera.yaml").read_text())["R_CB"],
        "phases": [{"name": n, "seconds": s} for n, s, _ in PHASES],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{i} frames ({i / (time.monotonic() - t0):.1f} fps) -> {out}")


if __name__ == "__main__":
    main()
