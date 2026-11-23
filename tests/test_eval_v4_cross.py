"""Checks for tools/eval_depth_wrist_v4_cross.py (M9) and the rotation limit of v4 it measures. NumPy only."""
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_depth_wrist_v3 import K, W, H, place, pnp, render, rot
from vision.depth_wrist_v3 import WristDepthV3
from vision.depth_wrist_v4 import GAIN, WristDepthV4, hand_span

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_depth_wrist_v4_cross as cross  # noqa: E402


def image_of(cam):
    pix = cam @ K.T
    pix = pix[:, :2] / pix[:, 2:]
    return np.column_stack([pix / (W, H), np.zeros(21)])


def test_angle_deg_and_nan():
    assert cross.angle_deg(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(90.0)
    assert cross.angle_deg(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) == pytest.approx(0.0)
    assert np.isnan(cross.angle_deg(np.full(3, np.nan), np.array([1.0, 0, 0])))


def test_classify_labels():
    base = {"angle_sensor_max_deg": 5.0, "angle_sensor_end_deg": 4.0, "flex_log_end": 0.0, "span_jitter_log": 0.01}
    assert cross.classify(base) == ["other"]
    assert cross.classify({**base, "angle_sensor_end_deg": 25.0}) == ["rotation"]
    assert cross.classify({**base, "angle_sensor_max_deg": None, "angle_sensor_end_deg": None, "flex_log_end": -0.2,
                           "span_jitter_log": 0.2}) == ["finger_flexion", "detection_jitter"]


def test_gain_variants_are_named_and_g0_is_listed():
    assert cross.V4 == f"g{GAIN:g}" and "g0" in cross.VARIANTS and cross.VARIANTS[0] == "v3"


# (axis from landmark a to b, degrees): along the hand (span -6%), across the palm both ways (-26%, +12%)
@pytest.mark.parametrize("axis, deg", (((0, 9), -30.0), ((17, 5), -30.0), ((17, 5), 30.0)))
def test_a_turn_in_place_moves_v4s_held_wrist_but_not_v3s(axis, deg):
    """The limit M9 measures: the wrist does not move, the palm turns 30 degrees, the span changes, v4 moves the range."""
    cam, image = place()
    depth = render(cam)
    turned = (cam - cam[0]) @ rot(cam[axis[1]] - cam[axis[0]], deg).T + cam[0]  # about the wrist: it stays put
    image2 = image_of(turned)
    v3, v4 = WristDepthV3(), WristDepthV4()
    v3(pnp(cam), image, None, depth, K, "right")
    s = v4(pnp(cam), image, None, depth, K, "right")
    for _ in range(40):
        h3 = v3(pnp(turned), image2, None, None, K, "right")
        h4 = v4(pnp(turned), image2, None, None, K, "right")
    assert s.source == "depth"
    truth, anchor = np.linalg.norm(cam[0]), np.linalg.norm(s.position)
    assert abs(anchor - truth) < 0.001  # the rendered sensor (mm steps) saw the wrist where it is
    assert abs(np.linalg.norm(h3.position) - anchor) < 1e-9  # v3 stays there: right, the wrist did not move
    expect = (hand_span(image, (W, H)) / hand_span(image2, (W, H))) ** GAIN
    assert h4.gap_factor == pytest.approx(expect, rel=1e-6)
    assert abs(np.log(h4.gap_factor)) > 0.01  # the turn alone moves v4 by more than 1% of the range
    assert abs(np.linalg.norm(h4.position) - truth) > 0.004
