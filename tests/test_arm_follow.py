"""Checks for arm/follow.py: camera-to-base axes, relative clutch mapping, and step limiting."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from arm.follow import R_CB, RelativeFollower, limit_step


def test_no_wrist_motion_keeps_engaged_eef_pose():
    f = RelativeFollower()
    w_p0 = np.array([0.05, -0.02, 0.55])
    w_R0 = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    e_p0 = np.array([0.37, 0.01, 0.35])
    e_R0 = Rotation.from_euler("xyz", [3.14, 0.1, -0.2]).as_matrix()

    f.engage(w_p0, w_R0, e_p0, e_R0)
    p, R = f.target(w_p0, w_R0)

    np.testing.assert_allclose(p, e_p0, atol=1e-12)
    np.testing.assert_allclose(R, e_R0, atol=1e-12)


@pytest.mark.parametrize(
    "cam_delta, base_delta",
    [
        (np.array([0.0, 0.0, -0.1]), np.array([0.1, 0.0, 0.0])),  # cam -z -> base +x
        (np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0])),   # cam +x -> base +y
        (np.array([0.0, -0.1, 0.0]), np.array([0.0, 0.0, 0.1])),  # cam -y -> base +z
    ],
)
def test_camera_translation_axes_map_to_base_axes(cam_delta, base_delta):
    f = RelativeFollower()
    w_p0 = np.array([0.1, 0.2, 0.6])
    w_R0 = np.eye(3)
    e_p0 = np.array([0.35, 0.0, 0.30])
    e_R0 = np.eye(3)

    f.engage(w_p0, w_R0, e_p0, e_R0)
    p, R = f.target(w_p0 + cam_delta, w_R0)

    np.testing.assert_allclose(p - e_p0, base_delta, atol=1e-12)
    np.testing.assert_allclose(R, e_R0, atol=1e-12)


def test_wrist_rotation_about_cam_z_rotates_target_about_base_minus_x():
    f = RelativeFollower()
    w_p0 = np.zeros(3)
    w_R0 = Rotation.from_euler("xyz", [0.3, -0.1, 0.5]).as_matrix()
    e_p0 = np.array([0.37, 0.0, 0.35])
    e_R0 = Rotation.from_euler("xyz", [0.4, 0.2, -0.3]).as_matrix()

    angle = 0.35
    rot_cam_z = Rotation.from_rotvec(np.array([0.0, 0.0, angle])).as_matrix()
    rot_base_minus_x = Rotation.from_rotvec(np.array([-angle, 0.0, 0.0])).as_matrix()

    f.engage(w_p0, w_R0, e_p0, e_R0)
    p, R = f.target(w_p0, rot_cam_z @ w_R0)

    np.testing.assert_allclose(p, e_p0, atol=1e-12)
    np.testing.assert_allclose(R, rot_base_minus_x @ e_R0, atol=1e-12)


def test_scale_multiplies_translation_only():
    f1 = RelativeFollower(scale=1.0)
    f2 = RelativeFollower(scale=2.5)
    w_p0 = np.array([0.0, 0.0, 0.5])
    w_R0 = np.eye(3)
    e_p0 = np.array([0.35, 0.0, 0.30])
    e_R0 = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()

    w_p1 = w_p0 + np.array([0.04, -0.03, -0.05])
    w_R1 = Rotation.from_euler("xyz", [0.2, -0.1, 0.15]).as_matrix()

    f1.engage(w_p0, w_R0, e_p0, e_R0)
    f2.engage(w_p0, w_R0, e_p0, e_R0)
    p1, R1 = f1.target(w_p1, w_R1)
    p2, R2 = f2.target(w_p1, w_R1)

    np.testing.assert_allclose(p2 - e_p0, 2.5 * (p1 - e_p0), atol=1e-12)
    np.testing.assert_allclose(R2, R1, atol=1e-12)


def test_workspace_clamp_bounds_target_position():
    lo = np.array([0.20, -0.15, 0.10])
    hi = np.array([0.45, 0.15, 0.50])
    f = RelativeFollower(R_cb=R_CB, scale=1.0, workspace_lo=lo, workspace_hi=hi)
    f.engage(np.zeros(3), np.eye(3), np.array([0.35, 0.0, 0.30]), np.eye(3))

    p, _ = f.target(np.array([0.50, 0.50, -0.50]), np.eye(3))
    np.testing.assert_allclose(p, np.array([0.45, 0.15, 0.10]), atol=1e-12)


def test_limit_step_clips_per_joint():
    q_prev = np.zeros(7)
    q_new = np.array([0.05, -0.20, 0.10, 0.50, -0.01, 0.12, -0.30])
    dt = 0.05
    max_vel = 2.0  # max step = 0.10 rad

    q_cmd = limit_step(q_prev, q_new, max_vel, dt)
    expected = np.array([0.05, -0.10, 0.10, 0.10, -0.01, 0.10, -0.10])
    np.testing.assert_allclose(q_cmd, expected, atol=1e-12)
