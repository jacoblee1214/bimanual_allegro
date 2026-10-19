#!/usr/bin/env python3
"""
measure_camera_tilt.py — the ego camera's tilt from its IMU, and the camera -> robot mapping.

For the egocentric setup (docs/05_ego_camera.md): a RealSense fixed above and behind the
operator's head, looking down at the hands the way a humanoid's head camera would. The
operator's motion must then reach the robot as seen from that head:
  robot x = forward, level with the floor     (the camera's viewing direction, levelled)
  robot y = left
  robot z = up                                 (against gravity)
This averages the accelerometer for a couple of seconds, turns it into the colour
camera's frame with the device's own IMU -> colour extrinsics, and writes the rotation
R_CB (camera_optical -> robot) that arm/follow.py's clutch mapping takes, plus the
colour intrinsics, to config/ego_camera.yaml. The intrinsics also go to
config/ego_camera_info.yaml in the ROS camera_info format vision/camera.py reads, so a
recorded ego clip can be replayed with the camera's own calibration.

An accelerometer at rest reads the reaction to gravity, which points up. Pitch is how
far the camera looks below the horizon, roll how far it is turned about its viewing axis.

Usage (with .venv-vision, camera still):
    tools/measure_camera_tilt.py [--seconds 2] [--out config/ego_camera.yaml]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision import camera  # noqa: E402


def mapping_from_up(up_cam: np.ndarray) -> np.ndarray:
    """R_CB whose rows are the robot's x, y, z axes written in the camera optical frame."""
    z = up_cam / np.linalg.norm(up_cam)
    view = np.array([0.0, 0.0, 1.0])                # camera optical z: where it looks
    x = view - z * (view @ z)                       # levelled viewing direction
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                              # left, for x forward and z up
    return np.vstack([x, y, z])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out", type=Path, default=REPO / "config" / "ego_camera.yaml")
    args = p.parse_args()

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.accel)
    profile = pipe.start(cfg)
    try:
        color = profile.get_stream(rs.stream.color).as_video_stream_profile()
        accel = profile.get_stream(rs.stream.accel)
        ext = accel.get_extrinsics_to(color)
        R_ac = np.array(ext.rotation, dtype=np.float64).reshape(3, 3).T   # librealsense stores column-major
        samples, t0 = [], time.time()
        while time.time() - t0 < args.seconds:
            for f in pipe.wait_for_frames(3000):
                if f.is_motion_frame():
                    a = f.as_motion_frame().get_motion_data()
                    samples.append([a.x, a.y, a.z])
        intr = color.get_intrinsics()
        device = profile.get_device().get_info(rs.camera_info.name)
    finally:
        pipe.stop()

    acc = np.array(samples)
    up_cam = R_ac @ acc.mean(axis=0)
    R_cb = mapping_from_up(up_cam)
    down = -up_cam / np.linalg.norm(up_cam)
    pitch = np.degrees(np.arcsin(np.clip(down[2], -1, 1)))       # + = looking down
    roll = np.degrees(np.arctan2(-down[0], down[1]))
    spread = np.degrees(np.arccos(np.clip((acc / np.linalg.norm(acc, axis=1, keepdims=True)) @
                                          (acc.mean(0) / np.linalg.norm(acc.mean(0))), -1, 1)).max())

    out = {
        "device": device,
        "measured": time.strftime("%Y-%m-%d %H:%M"),
        "pitch_down_deg": round(float(pitch), 2),
        "roll_deg": round(float(roll), 2),
        "gravity_m_s2": round(float(np.linalg.norm(acc.mean(0))), 3),
        "imu_samples": len(acc),
        "R_CB": [[round(float(v), 6) for v in row] for row in R_cb],
        "color": {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                  "cx": intr.ppx, "cy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("# Written by tools/measure_camera_tilt.py; re-run whenever the camera is moved.\n"
                        + yaml.safe_dump(out, sort_keys=False))
    print(f"{device}: pitch {pitch:+.1f} deg (down), roll {roll:+.1f} deg, |g| {np.linalg.norm(acc.mean(0)):.2f}, "
          f"{len(acc)} samples, spread {spread:.2f} deg")
    print("R_CB (rows = robot x forward, y left, z up, in camera_optical):")
    print(np.round(R_cb, 4))
    print(f"colour {intr.width}x{intr.height}: fx {intr.fx:.1f} fy {intr.fy:.1f} cx {intr.ppx:.1f} cy {intr.ppy:.1f}")
    info = args.out.with_name(args.out.stem + "_info.yaml")
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    # Distortion left out: the colour stream's model is inverse Brown-Conrady, the other way
    # round from OpenCV's, and its coefficients are small (see vision/hand_tracker_node_v2.py).
    camera.save(camera.CameraCalibration((intr.width, intr.height), K, np.zeros(5)), info, device,
                note=f"Factory intrinsics of the {device} colour stream, read by tools/measure_camera_tilt.py. "
                     f"Its distortion ({intr.model}, {np.round(intr.coeffs, 4).tolist()}) is left out.")
    print(f"-> {args.out}, {info}")


if __name__ == "__main__":
    main()
