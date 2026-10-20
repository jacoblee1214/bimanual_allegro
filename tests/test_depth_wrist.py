"""Synthetic checks for vision/depth_wrist.py: a rendered depth image, no camera. Runs in both venvs."""
import numpy as np

from vision.depth_filter import palm_frame
from vision.depth_wrist import (
    MIN_HOLD,
    PALM,
    DepthFusion,
    PalmDepth,
    depth_scale,
    hand_in_camera,
    palm_depth,
    polygon_pixels,
)

W, H = 640, 480
K = np.array([[383.7, 0.0, 321.4], [0.0, 383.4, 245.0], [0.0, 0.0, 1.0]])
BACKGROUND_M = 1.5

# Right hand in its own frame, metres (tests/test_wrist_pose.py), with the thumb CMC
# (1) moved into the palm plane so that the palm is exactly flat: the rendered sensor
# depth is then the plane the fusion compares against.
RIGHT_HAND = np.array([
    [0.000,  0.000, 0.000],
    [0.000,  0.020, 0.025], [0.020,  0.040, 0.045], [0.030,  0.055, 0.065], [0.040,  0.065, 0.080],
    [0.000,  0.022, 0.090], [0.010,  0.025, 0.130], [0.020,  0.026, 0.152], [0.030,  0.027, 0.170],
    [0.000,  0.000, 0.092], [0.010,  0.000, 0.135], [0.020,  0.000, 0.160], [0.030,  0.000, 0.180],
    [0.000, -0.020, 0.090], [0.010, -0.022, 0.128], [0.020, -0.023, 0.150], [0.030, -0.024, 0.168],
    [0.000, -0.038, 0.082], [0.010, -0.042, 0.112], [0.018, -0.044, 0.128], [0.025, -0.045, 0.142],
])


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])


# Palm towards the camera (canonical x to camera -z), fingers to image right, then tilted 30 degrees.
R_HAND = rot_y(30.0) @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
WRIST = np.array([-0.04, 0.02, 0.50])


def place(r=R_HAND, t=WRIST):
    """True hand in the camera frame and what MediaPipe would report for it."""
    cam = RIGHT_HAND @ r.T + t
    pix = cam @ K.T
    pix = pix[:, :2] / pix[:, 2:]
    image = np.column_stack([pix / (W, H), np.zeros(21)])
    r_world = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    world = (RIGHT_HAND - RIGHT_HAND.mean(axis=0)) @ r_world.T
    return cam, image, world


def render(cam, grow=1.4):
    """Depth image, metres at mm resolution: the true palm plane over a polygon grown past the
    palm landmarks (the palm reaches past its joint centres), a flat background elsewhere."""
    depth = np.full((H, W), BACKGROUND_M)
    palm = cam[list(PALM)]
    centre, normal = palm.mean(axis=0), np.linalg.svd(palm - palm.mean(axis=0))[2][-1]
    pix = palm @ K.T
    poly = pix[:, :2] / pix[:, 2:]
    poly = poly.mean(axis=0) + grow * (poly - poly.mean(axis=0))
    uv = polygon_pixels(poly, (H, W))
    rays = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))])
    depth[uv[:, 1], uv[:, 0]] = (centre @ normal) / (rays @ normal)
    return np.round(depth * 1000.0) / 1000.0, uv


def pnp_output(cam, scale_error):
    """The wrist pose PnP returns when MediaPipe's hand is `scale_error` times the real one."""
    wrong = cam * scale_error
    return wrong[0], palm_frame(wrong, "right")


def test_polygon_pixels_counts_a_square_and_clips_at_the_border():
    square = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    assert len(polygon_pixels(square - 0.5, (H, W))) == 100
    assert len(polygon_pixels(square - 15.5, (H, W))) == 25  # only u, v in 0..4 are in the image
    assert len(polygon_pixels(square + 1000.0, (H, W))) == 0


def test_palm_depth_reads_the_palm_not_the_background():
    cam, image, _ = place()
    depth, _ = render(cam)
    sensor = palm_depth(depth, image)
    assert sensor is not None
    assert sensor.fill == 1.0 and sensor.n > 300
    assert np.all(sensor.depths < 0.6)  # no background pixel kept
    assert abs(sensor.z - np.median(cam[list(PALM), 2])) < 0.02


def test_palm_depth_ignores_holes_saturated_pixels_and_stray_matches():
    cam, image, _ = place()
    clean, _ = render(cam)
    reference = palm_depth(clean, image).z
    noisy = clean.copy()
    rng = np.random.default_rng(1)
    pixels = palm_depth(clean, image).pixels
    pick = rng.permutation(len(pixels))
    holes, stray, saturated = np.split(pick[: int(0.55 * len(pick))], [int(0.4 * len(pick)), int(0.5 * len(pick))])
    noisy[pixels[holes, 1], pixels[holes, 0]] = 0.0
    noisy[pixels[stray, 1], pixels[stray, 0]] = 0.30         # stray matches near the minimum range
    noisy[pixels[saturated, 1], pixels[saturated, 0]] = 65.535  # failed pixels reported as 65535 counts
    sensor = palm_depth(noisy, image)
    assert sensor is not None
    assert abs(sensor.z - reference) < 0.003
    assert np.all(sensor.depths > 0.35) and np.all(sensor.depths < 0.6)
    assert 0.5 < sensor.fill < 0.7


def test_palm_depth_drops_background_seen_through_part_of_the_polygon():
    cam, image, _ = place()
    depth, _ = render(cam)
    reference = palm_depth(depth, image)
    corner = reference.pixels[reference.pixels[:, 0] < np.percentile(reference.pixels[:, 0], 25)]
    depth[corner[:, 1], corner[:, 0]] = BACKGROUND_M
    sensor = palm_depth(depth, image)
    assert sensor is not None
    assert np.all(sensor.depths < 0.6)
    assert abs(sensor.z - reference.z) < 0.01


def test_palm_depth_stays_off_the_polygon_edge():
    # Mixed depths at the hand's edge, too close to the palm for the MAD band to drop:
    # only the shrunk polygon keeps them out.
    cam, image, _ = place()
    _, inner_uv = render(cam, grow=0.9)
    depth, outer_uv = render(cam, grow=1.4)
    edge = np.zeros((H, W), bool)
    edge[outer_uv[:, 1], outer_uv[:, 0]] = True
    edge[inner_uv[:, 1], inner_uv[:, 0]] = False
    depth[edge] += 0.02
    sensor = palm_depth(depth, image)
    assert sensor is not None
    assert not edge[sensor.pixels[:, 1], sensor.pixels[:, 0]].any()


def test_palm_depth_is_none_without_enough_valid_pixels():
    cam, image, _ = place()
    depth, _ = render(cam)
    sparse = depth.copy()
    sparse[np.random.default_rng(2).random(depth.shape) < 0.8] = 0.0  # 20% valid, under MIN_FILL
    assert palm_depth(sparse, image) is None
    assert palm_depth(np.zeros_like(depth), image) is None
    off_image = image.copy()
    off_image[:, 0] += 2.0
    assert palm_depth(depth, off_image) is None


def test_hand_in_camera_gives_back_the_placed_hand():
    cam, _, world = place()
    hand = hand_in_camera(cam[0], palm_frame(cam, "right"), world, "right")
    np.testing.assert_allclose(hand, cam, atol=1e-12)


def test_scale_correction_recovers_the_true_wrist():
    cam, image, world = place()
    depth, _ = render(cam)
    sensor = palm_depth(depth, image)
    for error in (1 / 1.15, 1.15):
        p, r = pnp_output(cam, error)
        fused = DepthFusion()(p, r, world, sensor, K, "right")
        assert fused.source == "depth"
        assert abs(fused.scale * error - 1.0) < 0.004
        assert np.linalg.norm(fused.position - cam[0]) < 0.002
        # only the distance changes: the corrected wrist is on PnP's ray
        np.testing.assert_allclose(np.cross(fused.position, p), 0.0, atol=1e-12)


def test_scale_is_unbiased_when_half_of_a_tilted_palm_has_no_depth():
    cam, image, world = place(r=rot_y(25.0) @ R_HAND)  # 55 degrees from facing the camera
    depth, _ = render(cam)
    full = palm_depth(depth, image)
    far_side = full.pixels[full.depths > full.z]
    depth[far_side[:, 1], far_side[:, 0]] = 0.0
    half = palm_depth(depth, image, min_fill=0.2)
    assert half is not None and abs(half.z - full.z) > 0.01  # the median depth itself moves
    p, r = pnp_output(cam, 0.9)
    hand = hand_in_camera(p, r, world, "right")
    assert abs(depth_scale(half, hand, K) * 0.9 - 1.0) < 0.004


def test_edge_on_palm_gives_no_scale():
    # A palm in the plane x = 0 contains the ray of every pixel with u = cx.
    hand = place()[0].copy()
    n = len(PALM)
    hand[list(PALM)] = np.column_stack([np.zeros(n), np.linspace(-0.05, 0.05, n), 0.5 + 0.03 * np.sin(np.arange(n))])
    on_plane = PalmDepth(z=0.5, n=3, spread=0.0, fill=1.0,
                         pixels=np.array([[K[0, 2], 200.0], [K[0, 2], 240.0], [K[0, 2], 280.0]]), depths=np.full(3, 0.5))
    assert depth_scale(on_plane, hand, K) is None


def test_gate_falls_back_to_pnp_then_to_the_held_scale():
    cam, image, world = place()
    sensor = palm_depth(render(cam)[0], image)
    p, r = pnp_output(cam, 1 / 1.15)
    fusion = DepthFusion()

    first = fusion(p, r, world, None, K, "right")
    assert first.source == "pnp" and np.array_equal(first.position, p)

    for _ in range(3):
        assert fusion(p, r, world, sensor, K, "right").source == "depth"
    missing = fusion(p, r, world, None, K, "right")
    assert missing.source == "hold" and abs(missing.scale - 1.15) < 0.004
    assert np.linalg.norm(missing.position - cam[0]) < 0.002


def test_gate_rejects_scales_outside_the_absolute_band():
    cam, image, world = place()
    sensor = palm_depth(render(cam)[0], image)
    p, r = pnp_output(cam, 0.4)  # sensor / PnP = 2.5
    out = DepthFusion()(p, r, world, sensor, K, "right")
    assert out.source == "pnp" and abs(out.measured - 2.5) < 0.01


def test_gate_rejects_a_jump_away_from_recent_scales_but_never_locks_shut():
    cam, image, world = place()
    sensor = palm_depth(render(cam)[0], image)
    fusion = DepthFusion(window=10)
    for _ in range(MIN_HOLD):
        fusion(*pnp_output(cam, 1 / 1.15), world, sensor, K, "right")
    jump = fusion(*pnp_output(cam, 1 / 1.6), world, sensor, K, "right")  # 1.6: in the band, 39% off
    assert jump.source == "hold" and abs(jump.scale - 1.15) < 0.004

    # A wrong start: in-band scales near 0.6 first, then the hand's true 1.15 for good.
    fusion = DepthFusion(window=10)
    for _ in range(MIN_HOLD):
        fusion(*pnp_output(cam, 1 / 0.6), world, sensor, K, "right")
    sources = [fusion(*pnp_output(cam, 1 / 1.15), world, sensor, K, "right").source for _ in range(10)]
    assert sources[0] == "hold" and sources[-1] == "depth"
