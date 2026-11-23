"""vision/wrist_diag.py (NumPy only, both venvs) and hand_tracker_node_v6.py's fusion choice (needs ROS and OpenCV)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_depth_wrist_v3 import K, WRIST, place, pnp, render
from vision.depth_wrist_v3 import WristDepthV3
from vision.depth_wrist_v4 import WristDepthV4
from vision.wrist_diag import diag_json

REPO = Path(__file__).resolve().parents[1]
HAS_ROS_VISION = (
    int(np.__version__.split(".")[0]) < 2
    and importlib.util.find_spec("rclpy") is not None
    and importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("mediapipe") is not None
)
needs_ros = pytest.mark.skipif(not HAS_ROS_VISION, reason="needs ROS (source /opt/ros/jazzy/setup.bash), OpenCV and MediaPipe (.venv-vision)")


def frames(fusion):
    """A sensor frame, then a frame without depth (a hold)."""
    cam, image = place()
    a = fusion(pnp(cam), image, None, render(cam), K, "right")
    b = fusion(pnp(cam), image, None, None, K, "right")
    return a, b


def test_diag_v4_has_gap_factor_only_on_hold_frames():
    depth, hold = frames(WristDepthV4())
    d, h = json.loads(diag_json(depth, 0.045)), json.loads(diag_json(hold, 0.045))
    assert d["source"] == "depth" and d["gap_factor"] is None
    assert h["source"] == "hold" and h["gap_factor"] == pytest.approx(1.0)
    assert h["sigma_ray_mm"] == 45.0


def test_diag_without_gap_factor_keeps_the_older_keys_only():
    for fused in frames(WristDepthV3()):
        body = json.loads(diag_json(fused, 0.005))
        assert "gap_factor" not in body
        assert set(body) == {"source", "scale", "measured", "pixels", "sigma_ray_mm"}


def test_v4_diag_is_v3s_plus_one_key():
    for a, b in zip(frames(WristDepthV3()), frames(WristDepthV4())):
        assert set(json.loads(diag_json(b, 0.0))) - set(json.loads(diag_json(a, 0.0))) == {"gap_factor"}


@needs_ros
def test_node_v6_fusion_choice():
    sys.path.insert(0, str(REPO))
    from vision.hand_tracker_node_v6 import fusion_factory
    assert fusion_factory("v4") is WristDepthV4
    assert fusion_factory("v3") is WristDepthV3


@needs_ros
def test_node_v6_help_lists_v4_and_defaults_to_v1():
    out = subprocess.run([sys.executable, "-B", str(REPO / "vision" / "hand_tracker_node_v6.py"), "--help"],
                         capture_output=True, text=True, cwd=REPO, timeout=60)
    assert out.returncode == 0 and "{v1,v2,v3,v4}" in out.stdout


def test_real_ego_v2_script_picks_the_node_by_fusion():
    text = (REPO / "tools" / "run_real_ego_v2.sh").read_text()
    assert "v4) NODE=vision/hand_tracker_node_v6.py" in text
    assert "v2|v3) NODE=vision/hand_tracker_node_v5.py" in text
    assert "FUSION=v3" in text  # default unchanged
