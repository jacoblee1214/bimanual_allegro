"""
rgbd_source.py — colour frames with depth aligned to them: the RealSense, or a recorded clip.

Both give `read() -> (ok, bgr, depth_m)`, depth in metres as float32 on the colour
frame's pixel grid (0 = no data), or None when depth is off, and a `calibration`
(vision.camera.CameraCalibration) for the colour frame.

* `RealSenseRGBD`: colour 640x480 and depth 424x240, both at 30 fps, depth aligned to
  colour with rs.align. 424x240 because the D457's closest measurable distance is
  about 0.16 m there and 0.30 m at 640x480, where hands held near the body came out
  empty (tools/record_ego_depth.py, docs/results/m5_depth_wrist.md, 3). Intrinsics
  are the device's factory colour intrinsics with the distortion left out, as
  hand_tracker_node_v3.RealSenseColour does.

* `ClipRGBD`: a tools/record_ego_depth.py clip (color.avi, depth.npy, frames.jsonl,
  meta.json), so everything downstream can run without the camera. `t` is the
  recorded time of the frame just read (seconds since the recording started).

No ROS here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vision import camera

COLOUR_SIZE = (640, 480)
DEPTH_SIZE = (424, 240)
FPS = 30


def _calibration(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> camera.CameraCalibration:
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return camera.CameraCalibration((width, height), k, np.zeros(5))


class ClipRGBD:
    def __init__(self, clip: Path | str, depth: bool = True) -> None:
        self.clip = Path(clip)
        meta = json.loads((self.clip / "meta.json").read_text())
        c = meta["color"]
        self.calibration = _calibration(c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"])
        self.depth_scale = float(meta["depth_scale"])
        self.rows = [json.loads(line) for line in (self.clip / "frames.jsonl").open()]
        self.times = np.array([r["t"] for r in self.rows])
        self._depth = np.load(self.clip / "depth.npy", mmap_mode="r") if depth else None
        if self._depth is not None and len(self._depth) != len(self.rows):
            raise ValueError(f"{self.clip}: depth.npy has {len(self._depth)} rows, frames.jsonl {len(self.rows)}")
        self._video: Optional[cv2.VideoCapture] = None
        self.rewind()

    def rewind(self) -> None:
        if self._video is not None:
            self._video.release()
        self._video = cv2.VideoCapture(str(self.clip / "color.avi"))
        self.index = -1      # row of frames.jsonl just read
        self.t: Optional[float] = None

    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        if self.index + 1 >= len(self.rows):
            return False, None, None
        ok, frame = self._video.read()
        if not ok:
            return False, None, None
        self.index += 1
        self.t = float(self.times[self.index])
        depth = None
        if self._depth is not None:
            depth = self._depth[self.index].astype(np.float32) * np.float32(self.depth_scale)
        return True, frame, depth

    def release(self) -> None:
        self._video.release()


class RealSenseRGBD:
    """Colour and aligned depth; for colour alone, hand_tracker_node_v3.RealSenseColour."""

    def __init__(self, colour_size: tuple[int, int] = COLOUR_SIZE, depth_size: tuple[int, int] = DEPTH_SIZE,
                 fps: int = FPS) -> None:
        import pyrealsense2 as rs

        self._pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, colour_size[0], colour_size[1], rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps)
        profile = self._pipe.start(cfg)
        dev = profile.get_device()
        if not dev.get_info(rs.camera_info.usb_type_descriptor).startswith("3"):
            self._pipe.stop()
            sys.exit("camera is not on USB 3: colour + depth will not keep up (docs/05_ego_camera.md, 4)")
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # Inverse Brown-Conrady colour distortion, small; left out as in hand_tracker_node_v3.RealSenseColour.
        self.calibration = _calibration(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()
        self._align = rs.align(rs.stream.color)
        self.fps = fps

    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self._align.process(self._pipe.wait_for_frames(3000))
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c or not d:
            return False, None, None
        depth = np.asanyarray(d.get_data()).astype(np.float32) * np.float32(self.depth_scale)
        return True, np.asanyarray(c.get_data()).copy(), depth

    def release(self) -> None:
        self._pipe.stop()
