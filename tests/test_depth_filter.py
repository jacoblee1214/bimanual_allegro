"""Synthetic checks for vision/depth_filter.py: runs in both .venv-vision and .venv-arm."""
import numpy as np
import pytest

from vision.depth_filter import (
    RESET_GAP_S,
    HandSizeHold,
    RayKalman,
    is_back_facing,
    palm_frame as depth_filter_palm_frame,
    palm_size,
    unit_ray,
)

RATE = 30.0
DT = 1.0 / RATE

# Synthetic right hand in its own canonical frame (metres):
# x out of the palm, y towards the thumb, z along the middle finger.
RIGHT_HAND = np.array([
    [0.000,  0.000, 0.000],                                                              # 0: wrist
    [0.010,  0.020, 0.025], [0.020,  0.040, 0.045], [0.030,  0.055, 0.065], [0.040,  0.065, 0.080],  # 1-4: thumb
    [0.000,  0.022, 0.090], [0.010,  0.025, 0.130], [0.020,  0.026, 0.152], [0.030,  0.027, 0.170],  # 5-8: index
    [0.000,  0.000, 0.092], [0.010,  0.000, 0.135], [0.020,  0.000, 0.160], [0.030,  0.000, 0.180],  # 9-12: middle
    [0.000, -0.020, 0.090], [0.010, -0.022, 0.128], [0.020, -0.023, 0.150], [0.030, -0.024, 0.168],  # 13-16: ring
    [0.000, -0.038, 0.082], [0.010, -0.042, 0.112], [0.018, -0.044, 0.128], [0.025, -0.045, 0.142],  # 17-20: little
])
LEFT_HAND = RIGHT_HAND * (1.0, -1.0, 1.0)
HANDS = {"right": RIGHT_HAND, "left": LEFT_HAND}

# Rotate canonical hand so palm normal (+x in canonical) points along camera +z (back faces camera)
# or camera -z (palm faces camera).
R_BACK_TO_CAM = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
R_PALM_TO_CAM = np.array([[0.0, 0.0,  1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def test_measurement_covariance_is_elongated_along_the_camera_ray():
    p0 = np.array([0.12, -0.08, 0.60])
    u = unit_ray(p0)
    e_lat = np.cross(u, [0.0, 1.0, 0.0])
    e_lat /= np.linalg.norm(e_lat)

    kf = RayKalman(sigma_lat=0.002, sigma_ray=0.045)
    R = kf.measurement_covariance(p0)

    np.testing.assert_allclose(R @ u, (0.045 ** 2) * u, atol=1e-12)
    np.testing.assert_allclose(R @ e_lat, (0.002 ** 2) * e_lat, atol=1e-12)


def test_ray_noise_is_reduced_much_more_than_lateral_noise():
    p0 = np.array([0.12, -0.08, 0.60])
    u = unit_ray(p0)
    e_lat = np.cross(u, [0.0, 1.0, 0.0])
    e_lat /= np.linalg.norm(e_lat)

    rng = np.random.default_rng(0)
    n_samples = 400
    noise_ray = rng.normal(0.0, 0.020, n_samples)   # 20 mm along the ray
    noise_lat = rng.normal(0.0, 0.005, n_samples)   # 5 mm laterally

    kf = RayKalman()
    out = np.array([
        kf(p0 + noise_ray[i] * u + noise_lat[i] * e_lat, i * DT) - p0
        for i in range(n_samples)
    ])

    out_ray = out[50:] @ u
    out_lat = out[50:] @ e_lat
    in_ray = noise_ray[50:]
    in_lat = noise_lat[50:]

    ray_std_ratio = np.std(out_ray) / np.std(in_ray)
    lat_std_ratio = np.std(out_lat) / np.std(in_lat)
    assert ray_std_ratio < 0.45
    assert ray_std_ratio < 0.55 * lat_std_ratio

    ray_d2_ratio = np.median(np.abs(np.diff(out_ray, n=2))) / np.median(np.abs(np.diff(in_ray, n=2)))
    lat_d2_ratio = np.median(np.abs(np.diff(out_lat, n=2))) / np.median(np.abs(np.diff(in_lat, n=2)))
    assert ray_d2_ratio < 0.25 * lat_d2_ratio


def test_lateral_step_is_followed_within_a_few_frames():
    p0 = np.array([0.05, -0.04, 0.60])
    u = unit_ray(p0)
    e_lat = np.cross(u, [0.0, 1.0, 0.0])
    e_lat /= np.linalg.norm(e_lat)
    step = 0.050 * e_lat  # 50 mm lateral step

    kf = RayKalman()
    for i in range(20):
        kf(p0, i * DT)

    after_step = [kf(p0 + step, (20 + k) * DT) for k in range(6)]
    # Within 2-3 frames, lateral displacement reaches >90% of the 50 mm step.
    lat_err_frame2 = abs((after_step[1] - p0) @ e_lat - 0.050)
    assert lat_err_frame2 < 0.005  # < 5 mm (10% of step) by the 2nd frame after the step
    # And the ray (depth) coordinate does not jump toward the origin.
    ray_drift = max(abs((p - p0) @ u) for p in after_step)
    assert ray_drift < 0.005


def test_variable_dt_and_reset_after_a_gap():
    kf = RayKalman()
    p0 = np.array([0.0, 0.0, 0.55])
    # Variable dt sequence (between 15 ms and 55 ms)
    ts = np.cumsum([0.0, 0.020, 0.045, 0.018, 0.052, 0.030, 0.025])
    for t in ts:
        out = kf(p0, float(t))
        np.testing.assert_allclose(out, p0, atol=1e-12)

    # Gap > RESET_GAP_S (0.3 s) resets the filter to the new position immediately.
    p_new = np.array([0.25, -0.15, 0.75])
    jumped = kf(p_new, float(ts[-1] + RESET_GAP_S + 0.05))
    np.testing.assert_allclose(jumped, p_new, atol=1e-12)

    # Explicit reset() also forgets the state.
    kf(p0, float(ts[-1] + 0.40))
    kf.reset()
    np.testing.assert_allclose(kf(p_new, 10.0), p_new, atol=1e-12)


def test_hand_size_hold_makes_palm_size_constant():
    base = RIGHT_HAND @ R_BACK_TO_CAM.T
    target_size = palm_size(base)
    hold = HandSizeHold(window=15, back_only=False)

    # Establish reference scale then apply symmetric and spiky scale wobble.
    scales = [1.0, 1.0, 1.0, 1.12, 0.88, 1.08, 0.92, 1.15, 0.85, 1.05, 0.95, 1.20, 0.80]
    out_sizes = []
    for i, k in enumerate(scales):
        rescaled = hold(base * k, "right", i * DT)
        out_sizes.append(palm_size(rescaled))
        assert np.isclose(palm_size(rescaled), hold.held_size)

    # Whenever the running median equals the true scale (every odd step above), palm size is exact;
    # overall variation is reduced by more than an order of magnitude compared to input scales.
    in_sizes = [palm_size(base * k) for k in scales]
    assert np.std(out_sizes) < 0.1 * np.std(in_sizes)
    np.testing.assert_allclose(out_sizes[::2], target_size, atol=1e-12)

    # Gap > RESET_GAP_S resets the held palm size to the new hand's size.
    bigger = base * 1.25
    after_gap = hold(bigger, "right", 10.0)
    assert np.isclose(palm_size(after_gap), target_size * 1.25)


@pytest.mark.parametrize("side", ["right", "left"])
def test_hand_size_hold_back_only_scales_only_when_back_faces_camera(side):
    hand_back = HANDS[side] @ R_BACK_TO_CAM.T
    hand_palm = HANDS[side] @ R_PALM_TO_CAM.T
    assert is_back_facing(hand_back, side) is True
    assert is_back_facing(hand_palm, side) is False

    ref_size = palm_size(hand_back)
    hold = HandSizeHold(window=15, back_only=True, side=side)
    # Seed the median with the reference palm size.
    for i in range(5):
        hold(hand_back, side, i * DT)

    # Back facing camera: 15% scale inflation is removed so palm_size stays at ref_size.
    out_back = hold(hand_back * 1.15, side, 5 * DT)
    assert np.isclose(palm_size(out_back), ref_size)

    # Palm facing camera: landmarks pass through untouched.
    out_palm = hold(hand_palm * 1.15, side, 6 * DT)
    assert np.isclose(palm_size(out_palm), ref_size * 1.15)
    np.testing.assert_allclose(out_palm, hand_palm * 1.15, atol=1e-12)


def test_palm_frame_copy_matches_wrist_pose():
    """depth_filter carries a NumPy copy of palm_frame for .venv-arm; it must agree."""
    try:
        from vision.wrist_pose import palm_frame as original
    except ImportError:
        pytest.skip("vision.wrist_pose (OpenCV) does not import in this venv")
    rng = np.random.default_rng(0)
    for side in ("left", "right"):
        for _ in range(20):
            pts = rng.normal(0, 0.05, (21, 3))
            np.testing.assert_allclose(depth_filter_palm_frame(pts, side), original(pts, side), atol=1e-12)
