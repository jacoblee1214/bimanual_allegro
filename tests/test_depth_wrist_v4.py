"""Synthetic checks for vision/depth_wrist_v4.py: a rendered depth image, no camera. NumPy only, runs in both venvs."""
import numpy as np
import pytest

from tests.test_depth_wrist_v3 import K, WRIST, on_ray_at, place, pnp, render
from vision.depth_wrist_v3 import FusedWristV3, WristDepthV3
from vision.depth_wrist_v4 import STEP, FusedWristV4, WristDepthV4, hand_span

FRAMES = 40  # enough for the per-frame step limit to reach any target below


def sensor_then_gap(fusion, shift, frames=FRAMES, scales=None):
    """One sensor frame at WRIST, then `frames` frames without the sensor at WRIST + shift. Returns (sensor, last hold)."""
    cam, image = place()
    seen = fusion(pnp(cam), image, None, render(cam), K, "right")
    moved, image2 = place(WRIST + shift)
    out = None
    for i in range(frames):
        scale = 0.85 if scales is None else scales(i)
        out = fusion(pnp(moved, scale), image2, None, None, K, "right")
    return seen, out, moved


def test_sensor_frames_are_v3s_and_the_hold_without_motion_is_v3s():
    cam, image = place()
    depth = render(cam)
    v3, v4 = WristDepthV3(), WristDepthV4()
    a, b = v3(pnp(cam), image, None, depth, K, "right"), v4(pnp(cam), image, None, depth, K, "right")
    assert isinstance(b, FusedWristV4) and isinstance(b, FusedWristV3)
    assert b.source == "depth" and b.gap_factor is None
    np.testing.assert_array_equal(a.position, b.position)
    np.testing.assert_array_equal(a.rotation, b.rotation)
    assert (a.scale, a.measured, a.pixels) == (b.scale, b.measured, b.pixels)
    h3, h4 = v3(pnp(cam), image, None, None, K, "right"), v4(pnp(cam), image, None, None, K, "right")
    assert h4.source == "hold" and h4.rotation is None and h4.gap_factor == pytest.approx(1.0)
    np.testing.assert_allclose(h4.position, h3.position, atol=1e-12)


def test_gain_zero_is_v3_even_when_the_hand_moves():
    v3, v4 = WristDepthV3(), WristDepthV4(gain=0.0)
    _, h3, _ = sensor_then_gap(v3, (0.0, 0.0, 0.15))
    _, h4, _ = sensor_then_gap(v4, (0.0, 0.0, 0.15))
    np.testing.assert_allclose(h4.position, h3.position, atol=1e-12)


def test_hold_follows_the_hand_moving_away_and_towards_the_camera():
    for dz in (0.12, -0.10):
        seen, v3, moved = sensor_then_gap(WristDepthV3(), (0.0, 0.0, dz))
        _, v4, _ = sensor_then_gap(WristDepthV4(gain=1.0), (0.0, 0.0, dz))
        truth, d0 = np.linalg.norm(moved[0]), np.linalg.norm(seen.position)
        assert on_ray_at(v4.position, moved[0], np.linalg.norm(v4.position))
        assert abs(np.linalg.norm(v3.position) - d0) < 1e-9
        # v3 stays where the sensor last saw the hand; v4 (undamped) gets within 6% of the true range
        # (the hand has depth, so its span is not exactly inverse to the wrist's distance)
        assert abs(np.linalg.norm(v4.position) - truth) < 0.06 * truth < 0.5 * abs(d0 - truth)
        assert np.sign(v4.gap_factor - 1.0) == np.sign(dz)
        # damped by default: part of the way
        _, half, _ = sensor_then_gap(WristDepthV4(), (0.0, 0.0, dz))
        assert 0.3 < (np.linalg.norm(half.position) - d0) / (truth - d0) < 0.7


def test_the_range_moves_at_most_step_per_frame_and_at_most_max_rel():
    cam, image = place()
    fusion = WristDepthV4(gain=1.0)
    seen = fusion(pnp(cam), image, None, render(cam), K, "right")
    far, image2 = place(WRIST + (0.0, 0.0, 0.5))  # 0.9 m: span ratio far beyond max_rel
    ranges = [np.linalg.norm(fusion(pnp(far), image2, None, None, K, "right").position) for _ in range(FRAMES)]
    d0 = np.linalg.norm(seen.position)
    assert ranges[0] / d0 == pytest.approx(np.exp(STEP))
    assert np.all(np.diff(np.log(ranges)) <= STEP + 1e-9)
    assert ranges[-1] / d0 == pytest.approx(1.5)


def test_a_pnp_error_before_or_during_the_gap_does_not_move_the_held_wrist():
    shift = (0.02, 0.0, 0.10)
    cam, image = place()
    clean = WristDepthV4()
    _, ref, _ = sensor_then_gap(clean, shift)
    for weight in (0.0, 0.5):
        # the dry-run case: PnP 35% too close for 0.6 s before the gap, a different error in every gap frame
        fusion = WristDepthV4(pnp_weight=weight)
        depth = render(cam)
        for scale in [0.85] * 12 + [0.55] * 18:
            assert fusion(pnp(cam, scale), image, None, depth, K, "right").source == "depth"
        moved, image2 = place(WRIST + shift)
        for i in range(FRAMES):
            out = fusion(pnp(moved, (0.5, 1.5, 0.8)[i % 3]), image2, None, None, K, "right")
        assert out.source == "hold"
        np.testing.assert_allclose(out.position, ref.position, atol=1e-9)


def test_pnp_agreeing_with_the_span_is_mixed_in_and_one_that_disagrees_is_ignored():
    cam, image = place()
    depth = render(cam)
    shift = (0.0, 0.0, 0.10)
    moved, image2 = place(WRIST + shift)
    ratio = np.linalg.norm(moved[0]) / np.linalg.norm(cam[0])

    def last(weight, pnp_ratio):
        fusion = WristDepthV4(pnp_weight=weight)
        fusion(pnp(cam), image, None, depth, K, "right")
        for _ in range(FRAMES):
            out = fusion(pnp(moved, 0.85 * pnp_ratio / ratio), image2, None, None, K, "right")
        return out.gap_factor

    plain = last(0.0, ratio)
    # PnP says the hand went 10% further than the (damped) span does: within tolerance, so the mix moves
    assert last(0.5, plain * 1.10) > plain * 1.02
    # PnP says 60% further: ignored
    assert last(0.5, plain * 1.6) == pytest.approx(plain)


def test_sensor_return_resets_the_anchor_and_reset_goes_back_to_pnp():
    cam, image = place()
    fusion = WristDepthV4()
    pose = pnp(cam)
    first = fusion(pose, image, None, None, K, "right")
    assert first.source == "pnp" and np.array_equal(first.position, pose.position) and first.scale == 1.0
    assert first.gap_factor is None
    fusion(pose, image, None, render(cam), K, "right")
    moved, image2 = place(WRIST + (0.0, 0.0, 0.10))
    for _ in range(FRAMES):
        drifted = fusion(pnp(moved), image2, None, None, K, "right")
    assert drifted.gap_factor > 1.05
    # the sensor comes back at the new place: v3's/v2's wrist, and the next gap starts from there
    back = fusion(pnp(moved), image2, None, render(moved), K, "right")
    assert back.source == "depth" and back.gap_factor is None
    assert np.linalg.norm(back.position - moved[0]) < 0.002
    again = fusion(pnp(moved), image2, None, None, K, "right")
    assert again.source == "hold" and again.gap_factor == pytest.approx(1.0)
    assert np.linalg.norm(again.position - back.position) < 1e-9
    fusion.reset()
    assert fusion.held is None
    assert fusion(pose, image, None, None, K, "right").source == "pnp"
    # after reset a gap does not reuse the old anchor
    fusion(pose, image, None, render(cam), K, "right")
    assert fusion(pose, image, None, None, K, "right").gap_factor == pytest.approx(1.0)


def test_hand_span_scales_inversely_with_distance():
    _, near = place(WRIST)
    _, far = place(WRIST + (0.0, 0.0, 0.4))
    size = (640, 480)
    assert hand_span(near, size) / hand_span(far, size) == pytest.approx(2.0, rel=0.1)
