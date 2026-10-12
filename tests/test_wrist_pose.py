"""Synthetic-hand checks for vision/wrist_pose.py: no camera, no ROS."""
import cv2
import numpy as np
import pytest

from vision.wrist_pose import INDEX_MCP, WRIST, palm_frame, solve_wrist_pose

# A right hand in its own frame, metres: x out of the palm, y towards the thumb,
# z along the fingers. The fingers curl a little towards +x, the palm side.
# Middle MCP sits on z and index/ring MCPs differ only in y, so this hand's
# palm frame is exactly the identity.
RIGHT_HAND = np.array([
    [0.000,  0.000, 0.000],                                                              # wrist
    [0.010,  0.020, 0.025], [0.020,  0.040, 0.045], [0.030,  0.055, 0.065], [0.040,  0.065, 0.080],  # thumb
    [0.000,  0.022, 0.090], [0.010,  0.025, 0.130], [0.020,  0.026, 0.152], [0.030,  0.027, 0.170],  # index
    [0.000,  0.000, 0.092], [0.010,  0.000, 0.135], [0.020,  0.000, 0.160], [0.030,  0.000, 0.180],  # middle
    [0.000, -0.020, 0.090], [0.010, -0.022, 0.128], [0.020, -0.023, 0.150], [0.030, -0.024, 0.168],  # ring
    [0.000, -0.038, 0.082], [0.010, -0.042, 0.112], [0.018, -0.044, 0.128], [0.025, -0.045, 0.142],  # little
])
LEFT_HAND = RIGHT_HAND * (1.0, -1.0, 1.0)
HANDS = {"right": RIGHT_HAND, "left": LEFT_HAND}

IMAGE_SIZE = (1280, 720)
K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
WRIST_IN_CAMERA = np.array([0.10, -0.05, 0.60])


def rotation(seed):
    rvec = np.random.default_rng(seed).normal(size=3)
    return cv2.Rodrigues(rvec)[0]


def observe(hand, r_cam, t_cam, r_world):
    """What MediaPipe would report for `hand` placed at (r_cam, t_cam).

    World landmarks are centred on the hand and turned by an arbitrary
    `r_world`, since their axes are not something this module may rely on.
    """
    world = (hand - hand.mean(axis=0)) @ r_world.T
    cam = hand @ r_cam.T + t_cam
    uv = cam[:, :2] / cam[:, 2:] * (K[0, 0], K[1, 1]) + (K[0, 2], K[1, 2])
    image = np.column_stack([uv / IMAGE_SIZE, np.zeros(len(hand))])
    return world, image


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("seed", range(5))
def test_palm_frame_is_a_rotation_that_follows_the_hand(side, seed):
    r = rotation(seed)
    frame = palm_frame(HANDS[side] @ r.T, side)
    np.testing.assert_allclose(frame, r, atol=1e-12)
    assert np.isclose(np.linalg.det(frame), 1.0)


@pytest.mark.parametrize("side", ["right", "left"])
def test_palm_normal_points_out_of_the_palm_on_both_hands(side):
    hand = HANDS[side]
    x, y, _ = palm_frame(hand, side).T
    # Fingers curl towards the palm.
    assert np.dot(x, hand[8] - hand[INDEX_MCP]) > 0
    # y: thumb side on the right hand, little-finger side on the left.
    thumb = np.dot(y, hand[4] - hand[WRIST])
    assert thumb > 0 if side == "right" else thumb < 0


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("seed", range(5))
def test_exact_observation_recovers_the_pose(side, seed):
    r_cam = rotation(seed)
    world, image = observe(HANDS[side], r_cam, WRIST_IN_CAMERA, rotation(100 + seed))

    pose = solve_wrist_pose(world, image, IMAGE_SIZE, K, side)

    np.testing.assert_allclose(pose.position, WRIST_IN_CAMERA, atol=1e-6)
    np.testing.assert_allclose(pose.rotation, r_cam, atol=1e-6)
    assert pose.reprojection_px < 1e-3


def test_distance_scales_with_mediapipes_hand_size():
    """A hand MediaPipe sizes at 90% is placed at 90% of the distance (doc 3.3)."""
    world, image = observe(RIGHT_HAND, rotation(0), WRIST_IN_CAMERA, rotation(1))

    pose = solve_wrist_pose(world * 0.9, image, IMAGE_SIZE, K, "right")

    np.testing.assert_allclose(pose.position, WRIST_IN_CAMERA * 0.9, atol=1e-6)


@pytest.mark.parametrize("flat", [False, True], ids=["curled", "flat"])
def test_one_pixel_of_noise_stays_within_centimetres(flat):
    """Guards the solver choice: ITERATIVE lands up to 1.4 m off here (curled)."""
    hand = RIGHT_HAND * (0.0, 1.0, 1.0) if flat else RIGHT_HAND
    errors = []
    for seed in range(50):
        world, image = observe(hand, rotation(seed), WRIST_IN_CAMERA, rotation(100 + seed))
        noise = np.random.default_rng(seed).normal(0.0, 1.0, (21, 2)) / IMAGE_SIZE
        image[:, :2] += noise
        pose = solve_wrist_pose(world, image, IMAGE_SIZE, K, "right")
        errors.append(np.linalg.norm(pose.position - WRIST_IN_CAMERA))
    assert max(errors) < 0.02
