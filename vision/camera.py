"""
camera.py — camera intrinsics: checkerboard calibration and the YAML file that holds them.

The file is ROS's camera_info YAML (what camera_calibration writes and
camera_calibration_parsers reads), so a later camera_info publisher can load it
as is. Only the plumb_bob distortion model is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class CameraCalibration:
    image_size: tuple[int, int]   # (width, height) the intrinsics hold for
    camera_matrix: np.ndarray     # (3, 3)
    dist_coeffs: np.ndarray       # (5,) k1 k2 p1 p2 k3


def board_points(board: tuple[int, int]) -> np.ndarray:
    """(cols * rows, 3) inner-corner positions on the board, in squares.

    Square size is left at 1: it scales the board poses, not the intrinsics.
    """
    cols, rows = board
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return np.column_stack([grid, np.zeros(len(grid))]).astype(np.float32)


def calibrate(
    corner_sets: Sequence[np.ndarray], board: tuple[int, int], image_size: tuple[int, int]
) -> tuple[CameraCalibration, float]:
    """Intrinsics from detected inner corners, one (N, 1, 2) or (N, 2) array per view.

    Returns the calibration and its RMS reprojection error in pixels.
    """
    obj = board_points(board)
    rms, k, dist, _, _ = cv2.calibrateCamera(
        [obj] * len(corner_sets),
        [np.asarray(c, dtype=np.float32).reshape(-1, 1, 2) for c in corner_sets],
        image_size,
        None,
        None,
    )
    return CameraCalibration(tuple(image_size), k, dist.ravel()[:5]), float(rms)


def save(cal: CameraCalibration, path: Path | str, camera_name: str, note: str = "") -> None:
    w, h = cal.image_size
    k = cal.camera_matrix
    data = {
        "image_width": int(w),
        "image_height": int(h),
        "camera_name": camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in k.ravel()]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [float(v) for v in cal.dist_coeffs]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in np.eye(3).ravel()]},
        "projection_matrix": {
            "rows": 3, "cols": 4,
            "data": [float(v) for v in np.hstack([k, np.zeros((3, 1))]).ravel()],
        },
    }
    header = "".join(f"# {line}\n" for line in note.splitlines())
    Path(path).write_text(header + yaml.safe_dump(data, sort_keys=False))


def load(path: Path | str) -> CameraCalibration:
    data = yaml.safe_load(Path(path).read_text())
    if data["distortion_model"] != "plumb_bob":
        raise ValueError(f"{path}: only plumb_bob is supported, got {data['distortion_model']!r}")
    return CameraCalibration(
        image_size=(int(data["image_width"]), int(data["image_height"])),
        camera_matrix=np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3),
        dist_coeffs=np.array(data["distortion_coefficients"]["data"], dtype=np.float64),
    )
