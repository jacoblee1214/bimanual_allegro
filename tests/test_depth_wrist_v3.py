"""Synthetic checks for vision/depth_wrist_v3.py: a rendered depth image, no camera.

NumPy only, runs in both venvs; the one test through hand_tracker_ego_v3 needs OpenCV and is
skipped without it.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from vision.depth_filter import palm_frame
from vision.depth_wrist import PALM, palm_depth, polygon_pixels
from vision.depth_wrist_v2 import FusedWristV2, WristDepthV2
from vision.depth_wrist_v3 import FusedWristV3, WristDepthV3

HAS_TRACKER = False
if int(np.__version__.split(".")[0]) < 2:
    try:
        from vision.camera import CameraCalibration
        from vision.hand_tracker import RawHand
        from vision.hand_tracker_ego_v3 import SIGMA_RAY_SENSOR, DepthEgoHandTracker

        HAS_TRACKER = True
    except ImportError:
        HAS_TRACKER = False

W, H = 640, 480
K = np.array([[383.7, 0.0, 321.4], [0.0, 383.4, 245.0], [0.0, 0.0, 1.0]])
BACKGROUND_M = 1.5

# Right hand in its own frame, metres, palm exactly flat (as tests/test_depth_wrist_v2.py).
RIGHT_HAND = np.array([
    [0.000,  0.000, 0.000],
    [0.000,  0.020, 0.025], [0.020,  0.040, 0.045], [0.030,  0.055, 0.065], [0.040,  0.065, 0.080],
    [0.000,  0.022, 0.090], [0.010,  0.025, 0.130], [0.020,  0.026, 0.152], [0.030,  0.027, 0.170],
    [0.000,  0.000, 0.092], [0.010,  0.000, 0.135], [0.020,  0.000, 0.160], [0.030,  0.000, 0.180],
    [0.000, -0.020, 0.090], [0.010, -0.022, 0.128], [0.020, -0.023, 0.150], [0.030, -0.024, 0.168],
    [0.000, -0.038, 0.082], [0.010, -0.042, 0.112], [0.018, -0.044, 0.128], [0.025, -0.045, 0.142],
])


def rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * kx + (1.0 - np.cos(a)) * kx @ kx


# Back of the hand towards an egocentric camera, fingers up the image, tilted 40 degrees.
R_HAND = rot((1, 0, 0), 40.0) @ np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) @ palm_frame(RIGHT_HAND, "right").T
WRIST = np.array([0.03, 0.06, 0.40])


def place(t=WRIST):
    """True hand in the camera frame and the image landmarks MediaPipe would report for it."""
    cam = RIGHT_HAND @ R_HAND.T + t
    pix = cam @ K.T
    pix = pix[:, :2] / pix[:, 2:]
    return cam, np.column_stack([pix / (W, H), np.zeros(21)])


def render(cam, grow=1.4):
    """Depth image, metres at mm resolution: the true palm plane over the grown palm polygon, background elsewhere."""
    depth = np.full((H, W), BACKGROUND_M)
    palm = cam[list(PALM)]
    centre, normal = palm.mean(axis=0), np.linalg.svd(palm - palm.mean(axis=0))[2][-1]
    pix = palm @ K.T
    poly = pix[:, :2] / pix[:, 2:]
    poly = poly.mean(axis=0) + grow * (poly - poly.mean(axis=0))
    uv = polygon_pixels(poly, (H, W))
    rays = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))])
    depth[uv[:, 1], uv[:, 0]] = (centre @ normal) / (rays @ normal)
    return np.round(depth * 1000.0) / 1000.0


def pnp(cam, scale=0.85):
    """What PnP might return: the hand at the wrong distance."""
    return SimpleNamespace(position=cam[0] * scale, rotation=palm_frame(cam * scale, "right"), reprojection_px=0.0)


def on_ray_at(q, wrist, distance):
    """q lies on the ray through `wrist` at `distance` from the camera."""
    return (np.linalg.norm(np.cross(q, wrist)) / np.linalg.norm(wrist) < 1e-9
            and abs(np.linalg.norm(q) - distance) < 1e-9 and q @ wrist > 0)


def test_sensor_frames_are_v2s_wrist_and_palm_frame():
    cam, image = place()
    depth = render(cam)
    v2 = WristDepthV2()(pnp(cam), image, None, depth, K, "right")
    v3 = WristDepthV3()(pnp(cam), image, None, depth, K, "right")
    assert isinstance(v3, FusedWristV3) and isinstance(v3, FusedWristV2)
    assert v3.source == v2.source == "depth"
    np.testing.assert_array_equal(v3.position, v2.position)
    np.testing.assert_array_equal(v3.rotation, v2.rotation)
    assert v3.scale == v2.scale and v3.measured == v2.measured
    assert v3.pixels == palm_depth(depth, image).n >= 30


def test_hold_keeps_the_last_sensor_range_on_the_current_pixel_ray():
    cam, image = place()
    fusion = WristDepthV3()
    seen = fusion(pnp(cam), image, None, render(cam), K, "right")
    # the hand moves sideways and the sensor has nothing (no depth image at all, as without --depth)
    moved, image2 = place(WRIST + (0.06, -0.04, 0.0))
    out = fusion(pnp(moved), image2, None, None, K, "right")
    assert out.source == "hold" and out.rotation is None and out.pixels is None and out.measured is None
    assert on_ray_at(out.position, moved[0], np.linalg.norm(seen.position))
    assert out.scale == pytest.approx(np.linalg.norm(seen.position) / np.linalg.norm(moved[0] * 0.85))
    assert fusion.held == np.linalg.norm(seen.position)


def test_a_pnp_error_before_the_gap_does_not_move_the_held_wrist():
    # the dry-run case: PnP places the hand too close just before the sensor drops out; the
    # ratios go up but stay in the band, so v2's median follows them and its hold sits too far.
    cam, image = place()
    depth = render(cam)
    v2, v3 = WristDepthV2(), WristDepthV3()
    for scale in [0.85] * 12 + [0.55] * 18:  # 0.6 s of PnP 35% closer at 30 fps
        for f in (v2, v3):
            assert f(pnp(cam, scale), image, None, depth, K, "right").source == "depth"
    held2 = v2(pnp(cam), image, None, None, K, "right")
    held3 = v3(pnp(cam), image, None, None, K, "right")
    assert held2.source == held3.source == "hold"
    assert np.linalg.norm(held2.position) > 1.3 * np.linalg.norm(cam[0])
    assert abs(np.linalg.norm(held3.position) - np.linalg.norm(cam[0])) < 0.0015
    # and PnP's distance during the gap does not matter either
    for scale in (0.5, 0.85, 1.5):
        np.testing.assert_allclose(v3(pnp(cam, scale), image, None, None, K, "right").position, held3.position)


def test_out_of_band_and_sparse_sensor_frames_hold_and_report_what_the_sensor_had():
    cam, image = place()
    depth = render(cam)
    fusion = WristDepthV3()
    fusion(pnp(cam), image, None, depth, K, "right")
    out = fusion(pnp(cam, scale=0.4), image, None, depth, K, "right")
    assert out.source == "hold" and abs(out.measured - 2.5) < 0.02 and out.pixels == palm_depth(depth, image).n
    sparse = depth.copy()
    sparse[np.random.default_rng(2).random(sparse.shape) < 0.8] = 0.0
    out = fusion(pnp(cam), image, None, sparse, K, "right")
    assert out.source == "hold" and out.measured is None and out.pixels is None


def test_pnp_before_the_first_sensor_frame_and_after_reset():
    cam, image = place()
    fusion = WristDepthV3()
    pose = pnp(cam)
    first = fusion(pose, image, None, None, K, "right")
    assert first.source == "pnp" and np.array_equal(first.position, pose.position) and first.scale == 1.0
    assert first.rotation is None and fusion.held is None
    assert fusion(pose, image, None, render(cam), K, "right").source == "depth"
    assert fusion(pose, image, None, None, K, "right").source == "hold"
    fusion.reset()
    assert fusion(pose, image, None, None, K, "right").source == "pnp"


@pytest.mark.skipif(not HAS_TRACKER, reason="needs OpenCV (.venv-vision)")
def test_drops_into_the_ego_tracker_as_a_fusion_factory():
    cam, image = place()
    world = (RIGHT_HAND - RIGHT_HAND.mean(axis=0)) / 1.2  # MediaPipe's hand 20% small: PnP too close
    raw = RawHand(image, world, "Right", 0.9)
    frames = [[raw]] * 3

    def detector(rgb, t):
        return frames.pop(0)

    detector.close = lambda: None
    tracker = DepthEgoHandTracker(CameraCalibration((W, H), K, np.zeros(5)), fusion_factory=WristDepthV3,
                                  detector=detector)
    frame, depth = np.zeros((H, W, 3), np.uint8), render(cam)
    outs = [tracker.process(frame, i / 30.0, d)["right"] for i, d in enumerate((depth, depth, None))]
    assert [o.fused.source for o in outs] == ["depth", "depth", "hold"]
    assert outs[0].sigma_ray == SIGMA_RAY_SENSOR and outs[2].sigma_ray > SIGMA_RAY_SENSOR
    assert outs[0].fused.pixels >= 30
    for o in outs:
        assert np.linalg.norm(o.fused.position - cam[0]) < 0.002
