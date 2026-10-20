"""Unit tests for tools/analyze_hand_scale.py: synthetic hands and depth maps."""
import numpy as np
import pytest

HAS_CV2 = False
if int(np.__version__.split(".")[0]) < 2:
    try:
        import cv2  # noqa: F401
        from tools.analyze_hand_scale import (
            PALM_POLY,
            POSE_FEATURE_NAMES,
            evaluate_cross_session_pair,
            extract_pose_features,
            fit_linear_pose_model,
            palm_polygon_mask,
            pnp_palm_depth,
            predict_linear_pose_model,
            run_cross_session_evaluations,
            run_split_evaluation,
            sensor_palm_depth,
            solve_hand_in_camera,
        )

        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

pytestmark = pytest.mark.skipif(not HAS_CV2, reason="needs OpenCV (.venv-vision)")

IMAGE_SIZE = (640, 480)
K = np.array([[383.76, 0.0, 321.35], [0.0, 383.36, 245.01], [0.0, 0.0, 1.0]])

# Canonical right hand in metres: x out of palm, y towards thumb, z along middle finger.
RIGHT_HAND = np.array([
    [0.000,  0.000, 0.000],                                                              # 0: wrist
    [0.010,  0.020, 0.025], [0.020,  0.040, 0.045], [0.030,  0.055, 0.065], [0.040,  0.065, 0.080],  # 1-4: thumb
    [0.000,  0.022, 0.090], [0.010,  0.025, 0.130], [0.020,  0.026, 0.152], [0.030,  0.027, 0.170],  # 5-8: index
    [0.000,  0.000, 0.092], [0.010,  0.000, 0.135], [0.010,  0.000, 0.160], [0.010,  0.000, 0.180],  # 9-12: middle
    [0.000, -0.020, 0.090], [0.010, -0.022, 0.128], [0.020, -0.023, 0.150], [0.030, -0.024, 0.168],  # 13-16: ring
    [0.000, -0.038, 0.082], [0.010, -0.042, 0.112], [0.018, -0.044, 0.128], [0.025, -0.045, 0.142],  # 17-20: little
])
LEFT_HAND = RIGHT_HAND * (1.0, -1.0, 1.0)
HANDS = {"right": RIGHT_HAND, "left": LEFT_HAND}

R_BACK_TO_CAM = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
R_PALM_TO_CAM = np.array([[0.0, 0.0,  1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def observe_hand(hand_3d: np.ndarray, r_cam: np.ndarray, t_cam: np.ndarray, world_scale: float = 1.0):
    """Project true 3D hand at (r_cam, t_cam) and return (world_landmarks, image_landmarks, true_in_camera)."""
    cam = hand_3d @ r_cam.T + t_cam.reshape(1, 3)
    uv = cam[:, :2] / cam[:, 2:] * (K[0, 0], K[1, 1]) + (K[0, 2], K[1, 2])
    image = np.column_stack([uv / IMAGE_SIZE, np.zeros(len(hand_3d))])
    world = (hand_3d - hand_3d.mean(axis=0)) * world_scale
    return world, image, cam


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("true_scale", [0.88, 1.00, 1.15])
def test_pnp_and_sensor_recover_exact_scale_factor(side, true_scale):
    """If the real hand is `true_scale` times MediaPipe's world hand, s_true = z_sensor / z_pnp == true_scale."""
    t_cam = np.array([0.04, -0.02, 0.52])
    # Real hand has size `true_scale * HANDS[side]`, while MediaPipe reports unscaled `HANDS[side]`
    world, image, true_cam = observe_hand(
        HANDS[side] * true_scale, R_BACK_TO_CAM, t_cam, world_scale=1.0 / true_scale
    )
    solved = solve_hand_in_camera(world, image, IMAGE_SIZE, K)
    assert solved is not None
    in_cam, reproj = solved
    assert reproj < 1e-3
    np.testing.assert_allclose(in_cam * true_scale, true_cam, atol=1e-5)

    z_pnp = pnp_palm_depth(in_cam)
    true_palm_z = float(np.mean(true_cam[list(PALM_POLY), 2]))
    assert np.isclose(true_palm_z / z_pnp, true_scale, atol=1e-5)

    # Build synthetic uint16 depth map (depth_scale = 0.001 m) with constant palm depth = true_palm_z
    depth_scale = 0.001
    depth_u16 = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint16)
    mask = palm_polygon_mask(image, IMAGE_SIZE)
    depth_u16[mask == 1] = int(round(true_palm_z / depth_scale))

    d_info = sensor_palm_depth(depth_u16, image, depth_scale, IMAGE_SIZE)
    assert d_info["quality_ok"] is True
    assert np.isclose(d_info["z_sensor_m"], true_palm_z, atol=1e-3)
    s_meas = d_info["z_sensor_m"] / z_pnp
    assert np.isclose(s_meas, true_scale, atol=2e-3)


def test_sensor_palm_depth_rejects_background_leakage_and_flags_holes():
    t_cam = np.array([0.0, 0.0, 0.50])
    _, image, _ = observe_hand(RIGHT_HAND, R_BACK_TO_CAM, t_cam)
    mask = palm_polygon_mask(image, IMAGE_SIZE)
    ys, xs = np.nonzero(mask == 1)
    assert len(ys) > 100

    depth_scale = 0.001
    depth_u16 = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint16)
    depth_u16[ys, xs] = 500  # 0.500 m foreground palm
    # Corrupt 20% of polygon pixels with 1.35 m background leakage and 10% with 0 holes
    n = len(ys)
    depth_u16[ys[: int(0.20 * n)], xs[: int(0.20 * n)]] = 1350
    depth_u16[ys[int(0.20 * n) : int(0.30 * n)], xs[int(0.20 * n) : int(0.30 * n)]] = 0

    d_info = sensor_palm_depth(depth_u16, image, depth_scale, IMAGE_SIZE)
    assert d_info["quality_ok"] is True
    assert np.isclose(d_info["z_sensor_m"], 0.500, atol=1e-3)
    assert d_info["inlier_mad_m"] == 0.0

    # When 90% of the polygon is zero (e.g. hand closer than MinZ), quality_ok must be False
    mostly_hole = np.zeros_like(depth_u16)
    mostly_hole[ys[: int(0.08 * n)], xs[: int(0.08 * n)]] = 500
    d_hole = sensor_palm_depth(mostly_hole, image, depth_scale, IMAGE_SIZE, min_valid_fraction=0.20)
    assert d_hole["quality_ok"] is False


@pytest.mark.parametrize("side", ["left", "right"])
def test_pose_features_distinguish_orientation_curl_and_distance(side):
    t_near = np.array([0.0, 0.0, 0.45])
    t_far = np.array([0.0, 0.0, 0.70])
    w_back, img_back, cam_back = observe_hand(HANDS[side], R_BACK_TO_CAM, t_near)
    w_palm, img_palm, cam_palm = observe_hand(HANDS[side], R_PALM_TO_CAM, t_far)

    f_back = extract_pose_features(w_back, img_back, cam_back, IMAGE_SIZE, side)
    f_palm = extract_pose_features(w_palm, img_palm, cam_palm, IMAGE_SIZE, side)

    assert f_back["cos_normal_ray"] > 0.9
    assert f_palm["cos_normal_ray"] < -0.9
    assert f_back["apparent_size_px"] > f_palm["apparent_size_px"] * 1.4

    # Curl the 4 fingers (fold tips back towards MCPs)
    curled = HANDS[side].copy()
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        curled[dip] = curled[mcp] + np.array([0.035, 0.0, 0.020])
        curled[tip] = curled[mcp] + np.array([0.025, 0.0, 0.005])
    w_curl, img_curl, cam_curl = observe_hand(curled, R_BACK_TO_CAM, t_near)
    f_curl = extract_pose_features(w_curl, img_curl, cam_curl, IMAGE_SIZE, side)
    assert f_curl["finger_curl"] > f_back["finger_curl"] + 0.35


def test_split_evaluation_recovers_constant_and_linear_pose_scale():
    rng = np.random.default_rng(7)
    n = 120
    x_pose = rng.normal(0.0, 1.0, size=(n, 5))
    # True scale has a constant +12% offset plus a pose-dependent term on feature 0 and 1
    w_true = np.array([0.04, -0.025, 0.0, 0.0, 0.0])
    s_true = 1.12 + x_pose @ w_true + rng.normal(0.0, 0.005, size=n)
    z_pnp = rng.uniform(0.45, 0.65, size=n)
    z_sensor = s_true * z_pnp

    train_mask = np.arange(n) < 60
    test_mask = np.arange(n) >= 60
    res = run_split_evaluation(s_true, z_pnp, z_sensor, x_pose, train_mask, test_mask)
    assert res["valid_split"] is True

    te_raw = res["test"]["no_correction"]
    te_const = res["test"]["const_per_side"]
    te_lin = res["test"]["linear_pose_per_side"]

    assert te_const["depth_err_median_abs_mm"] < te_raw["depth_err_median_abs_mm"] * 0.35
    assert te_lin["depth_err_median_abs_mm"] < te_const["depth_err_median_abs_mm"] * 0.25
    assert te_lin["r2_variance_within_set"] > 0.90


def test_cross_session_evaluation_filters_quality_gate_and_transfers_constant():
    rng = np.random.default_rng(19)

    def make_side_records(true_scale: float, n_ok: int, n_bad: int) -> list[dict]:
        recs = []
        for i in range(n_ok + n_bad):
            ok = i < n_ok
            z_pnp = float(rng.uniform(0.30, 0.45))
            s = float(true_scale + rng.normal(0.0, 0.01)) if ok else 2.80
            feats = {k: float(rng.normal(0.0, 1.0)) for k in POSE_FEATURE_NAMES}
            recs.append(
                {
                    "frame": i,
                    "quality_ok": ok,
                    "s_true": s,
                    "z_pnp_m": z_pnp,
                    "z_sensor_m": s * z_pnp,
                    "features": feats,
                }
            )
        return recs

    clip_a = {"left": make_side_records(0.92, 40, 10), "right": make_side_records(1.14, 40, 10)}
    clip_b = {"left": make_side_records(0.92, 30, 8), "right": make_side_records(1.14, 30, 8)}

    pair = evaluate_cross_session_pair(clip_a, clip_b)
    assert pair["pooled_both_sides"]["valid"] is True
    for s, target in (("left", 0.92), ("right", 1.14)):
        sr = pair["sides"][s]
        assert sr["n_train"] == 40
        assert sr["n_test"] == 30
        assert np.isclose(sr["train_const_median"], target, atol=0.02)
        raw_med = sr["test"]["no_correction"]["depth_err_median_abs_mm"]
        const_med = sr["test"]["const_per_side"]["depth_err_median_abs_mm"]
        assert const_med < raw_med * 0.30

    all_cross = run_cross_session_evaluations({"clip_a": clip_a, "clip_b": clip_b})
    assert "clip_a_half1_to_half2" in all_cross
    assert "clip_a_to_clip_b" in all_cross
    assert "clip_b_to_clip_a" in all_cross

