"""
wrist_pose.py — wrist position and palm frame in the camera frame, from one RGB frame.

MediaPipe gives two views of the same 21 points: image landmarks (normalised
pixels, so where the hand is on screen) and world landmarks (metres, but with
the origin at the hand's own centre, so no distance to the camera). Pairing them
is a PnP problem, and its answer puts the hand in the camera frame.

No ROS here, so the maths can be checked against synthetic hands
(tests/test_wrist_pose.py) before a camera is involved. See
docs/01_vision_design.md, 3.3 and 3.4.

Inputs must come from the **unmirrored** camera image. A mirrored image
reverses the hand's chirality, and the palm frame below would point into the
back of the hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

N_LANDMARKS = 21

WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP = 0, 5, 9, 13


@dataclass(frozen=True)
class WristPose:
    position: np.ndarray      # (3,) wrist (landmark 0) in the camera optical frame, metres
    rotation: np.ndarray      # (3, 3) palm frame axes as columns, in the camera optical frame
    reprojection_px: float    # RMS reprojection error over the 21 points, pixels


def palm_frame(points: np.ndarray, side: str) -> np.ndarray:
    """(3, 3) palm frame axes as columns, in the frame `points` is given in.

    The same landmarks as `canonical_axes` in the MANUS project
    (`v6f_teleop/hand/mano.py`), so the arm and the retargeter share one hand
    frame:

        z  wrist to the middle MCP, along the hand
        x  palm normal, pointing out of the palm, for **both** hands
        y  z cross x: towards the thumb on a right hand, the little finger on a left

    `canonical_axes` builds y from the ring-to-index direction on either hand,
    which leaves x out of the palm on a right hand and out of the back on a
    left. That suits the retargeter, which compares a hand against a robot hand
    of the same side. An arm following the wrist is easier to reason about when
    x means the same thing on both sides, so the left hand's helper axis is
    reversed here.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    pts = np.asarray(points, dtype=np.float64).reshape(N_LANDMARKS, 3)

    z = pts[MIDDLE_MCP] - pts[WRIST]
    z = z / np.linalg.norm(z)
    y_aux = pts[INDEX_MCP] - pts[RING_MCP]
    if side == "left":
        y_aux = -y_aux
    x = np.cross(y_aux, z)
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def solve_wrist_pose(
    world_landmarks: np.ndarray,
    image_landmarks: np.ndarray,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    side: str,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[WristPose]:
    """Wrist pose in the camera optical frame (x right, y down, z forward).

    Args:
        world_landmarks: (21, 3) or (63,) MediaPipe world landmarks, metres.
        image_landmarks: (21, 3) or (63,) MediaPipe image landmarks; x and y
            are over image width and height, z is ignored.
        image_size: (width, height) of the frame MediaPipe saw, pixels.
        camera_matrix: (3, 3) intrinsics of that frame.
        side: "left" or "right", the operator's actual hand.

    Returns None if OpenCV finds no solution.

    All 21 points, SQPnP. Measured on synthetic hands with OpenCV 4.10
    (2026-10-12): the five palm points alone have no solver that holds up --
    SQPnP misses by 10-280 mm on them even with exact input, and ITERATIVE
    raises once they leave a plane (its DLT start needs six). With all 21,
    ITERATIVE's start is poor on a curled hand under 1 px of noise (p95 1.4 m);
    SQPnP stays at a 2 mm median, 7 mm p95, curled or flat.

    The distance comes out in proportion to MediaPipe's idea of the hand's
    size: a hand 10% bigger than it thinks is placed about 10% too close.
    """
    world = np.asarray(world_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    image = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    w, h = image_size
    pixels = image[:, :2] * (w, h)

    dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(world, pixels, camera_matrix, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None

    projected, _ = cv2.projectPoints(world, rvec, tvec, camera_matrix, dist)
    reprojection = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))

    r, _ = cv2.Rodrigues(rvec)
    in_camera = world @ r.T + tvec.reshape(1, 3)
    return WristPose(
        position=in_camera[WRIST],
        rotation=palm_frame(in_camera, side),
        reprojection_px=reprojection,
    )
