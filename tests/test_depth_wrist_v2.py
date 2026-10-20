"""Synthetic checks for vision/depth_wrist_v2.py: a rendered depth image, no camera. NumPy only, runs in both venvs."""
from types import SimpleNamespace

import numpy as np

from vision.depth_filter import palm_frame
from vision.depth_wrist import PALM, FusedWrist, palm_depth, polygon_pixels
from vision.depth_wrist_v2 import FusedWristV2, WristDepthV2, on_plane, sensor_plane

W, H = 640, 480
K = np.array([[383.7, 0.0, 321.4], [0.0, 383.4, 245.0], [0.0, 0.0, 1.0]])
BACKGROUND_M = 1.5

# Right hand in its own frame, metres, palm exactly flat (as tests/test_depth_wrist.py), so the
# rendered sensor surface is the plane of the joint centres.
RIGHT_HAND = np.array([
    [0.000,  0.000, 0.000],
    [0.000,  0.020, 0.025], [0.020,  0.040, 0.045], [0.030,  0.055, 0.065], [0.040,  0.065, 0.080],
    [0.000,  0.022, 0.090], [0.010,  0.025, 0.130], [0.020,  0.026, 0.152], [0.030,  0.027, 0.170],
    [0.000,  0.000, 0.092], [0.010,  0.000, 0.135], [0.020,  0.000, 0.160], [0.030,  0.000, 0.180],
    [0.000, -0.020, 0.090], [0.010, -0.022, 0.128], [0.020, -0.023, 0.150], [0.030, -0.024, 0.168],
    [0.000, -0.038, 0.082], [0.010, -0.042, 0.112], [0.018, -0.044, 0.128], [0.025, -0.045, 0.142],
])
LEFT_HAND = RIGHT_HAND * (1.0, -1.0, 1.0)  # mirror image: thumb on the other side


def rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * kx + (1.0 - np.cos(a)) * kx @ kx


# Back of the hand towards an egocentric camera (canonical x away from it), fingers up the
# image, then tilted 40 degrees about the image x axis: an oblique, nearly flat palm.
# Columns: where the hand's palm frame axes x, y, z go in the camera frame.
R_HAND = rot((1, 0, 0), 40.0) @ np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) @ palm_frame(RIGHT_HAND, "right").T
WRIST = np.array([0.03, 0.06, 0.40])


def place(hand=RIGHT_HAND, r=R_HAND, t=WRIST):
    """True hand in the camera frame and the image landmarks MediaPipe would report for it."""
    cam = hand @ r.T + t
    pix = cam @ K.T
    pix = pix[:, :2] / pix[:, 2:]
    return cam, np.column_stack([pix / (W, H), np.zeros(21)])


def render(cam, grow=1.4):
    """Depth image, metres at mm resolution: the true palm plane over the palm polygon grown past
    its landmarks, a flat background elsewhere."""
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


def pnp_pose(cam, side, scale=0.85, turn=None):
    """What PnP might return: the hand at the wrong distance and, with `turn`, in the wrong tilt
    (turned about the palm centre)."""
    hand = cam.copy()
    if turn is not None:
        c = hand[list(PALM)].mean(axis=0)
        hand = c + (hand - c) @ turn.T
    hand = hand * scale
    return SimpleNamespace(position=hand[0], rotation=palm_frame(hand, side), reprojection_px=0.0)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


def test_sensor_plane_recovers_the_palm_plane_facing_away_from_the_camera():
    for tilt in range(-50, 51, 10):  # SVD's sign of the normal is arbitrary; it must come out the same
        cam, image = place(r=rot((0, 1, 0), tilt) @ R_HAND)
        plane = sensor_plane(palm_depth(render(cam), image), K)
        palm = cam[list(PALM)]
        true_n = np.linalg.svd(palm - palm.mean(axis=0))[2][-1]
        assert abs(abs(plane.normal @ true_n) - 1.0) < 1e-4
        assert plane.normal @ plane.point > 0
        assert abs((plane.point - palm.mean(axis=0)) @ true_n) < 0.001
        assert plane.rms < 0.001


def test_wrist_is_on_its_pixel_ray_at_the_sensor_depth_whatever_pnp_says():
    cam, image = place()
    depth = render(cam)
    outs = []
    for pose in (pnp_pose(cam, "right"), pnp_pose(cam, "right", scale=1.2, turn=rot((1, 0, 0), -60.0))):
        out = WristDepthV2()(pose, image, None, depth, K, "right")
        assert out.source == "depth"
        assert np.linalg.norm(out.position - cam[0]) < 0.0015
        np.testing.assert_allclose(np.cross(out.position, cam[0]) / np.linalg.norm(cam[0]) ** 2, 0.0, atol=1e-9)
        outs.append(out)
    # the wrist does not depend on PnP's rotation or distance at all
    np.testing.assert_allclose(outs[0].position, outs[1].position, atol=1e-12)
    np.testing.assert_allclose(outs[0].rotation, outs[1].rotation, atol=1e-12)


def test_palm_frame_comes_from_the_sensor_for_either_hand():
    for hand, side in ((RIGHT_HAND, "right"), (LEFT_HAND, "left")):
        cam, image = place(hand)
        wrong = pnp_pose(cam, side, turn=rot((1, 0, 0), 50.0))
        assert angle_deg(wrong.rotation, palm_frame(cam, side)) > 45.0
        out = WristDepthV2()(wrong, image, None, render(cam), K, side)
        assert out.source == "depth"
        assert angle_deg(out.rotation, palm_frame(cam, side)) < 1.0
        assert out.rotation[:, 0] @ cam[0] > 0  # back of the hand to the camera: x points away


def test_output_is_still_a_v1_fused_wrist():
    cam, image = place()
    out = WristDepthV2()(pnp_pose(cam, "right"), image, None, render(cam), K, "right")
    assert isinstance(out, FusedWristV2) and isinstance(out, FusedWrist)
    assert abs(out.scale - out.measured) < 1e-12
    assert abs(out.scale - np.linalg.norm(cam[0]) / np.linalg.norm(cam[0] * 0.85)) < 0.005


def test_no_depth_falls_back_to_pnp_then_to_the_held_ratio_without_a_rotation():
    cam, image = place()
    depth = render(cam)
    pose = pnp_pose(cam, "right")
    fusion = WristDepthV2()
    first = fusion(pose, image, None, None, K, "right")
    assert first.source == "pnp" and np.array_equal(first.position, pose.position) and first.rotation is None
    for _ in range(3):
        assert fusion(pose, image, None, depth, K, "right").source == "depth"
    held = fusion(pose, image, None, None, K, "right")
    assert held.source == "hold" and held.rotation is None
    assert np.linalg.norm(held.position - cam[0]) < 0.002
    fusion.reset()
    assert fusion(pose, image, None, None, K, "right").source == "pnp"


def test_a_polygon_without_enough_depth_is_not_used():
    cam, image = place()
    sparse = render(cam)
    sparse[np.random.default_rng(2).random(sparse.shape) < 0.8] = 0.0
    assert WristDepthV2()(pnp_pose(cam, "right"), image, None, sparse, K, "right").source == "pnp"


def test_absolute_band_rejects_an_implausible_ratio():
    cam, image = place()
    out = WristDepthV2()(pnp_pose(cam, "right", scale=0.4), image, None, render(cam), K, "right")
    assert out.source == "pnp" and abs(out.measured - 2.5) < 0.02


def test_pnp_jumping_inside_the_band_does_not_take_the_sensor_away():
    # v1's relative band would hold here: the ratio jumps 40% because PnP did, not the sensor.
    cam, image = place()
    depth = render(cam)
    fusion = WristDepthV2()
    for _ in range(10):
        fusion(pnp_pose(cam, "right"), image, None, depth, K, "right")
    out = fusion(pnp_pose(cam, "right", scale=0.6), image, None, depth, K, "right")
    assert out.source == "depth" and np.linalg.norm(out.position - cam[0]) < 0.0015


def test_a_ray_lying_in_the_plane_gives_no_point():
    # a plane through the camera's y axis contains the ray of every pixel with u = cx
    plane = SimpleNamespace(point=np.array([0.0, 0.0, 0.5]), normal=np.array([1.0, 0.0, 0.0]))
    assert on_plane(plane, np.array([[K[0, 2], 200.0]]), K) is None
    tilted = SimpleNamespace(point=np.array([0.0, 0.0, 0.5]), normal=np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(on_plane(tilted, np.array([[K[0, 2], K[1, 2]]]), K), [[0.0, 0.0, 0.5]])
