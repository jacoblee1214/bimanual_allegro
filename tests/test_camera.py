"""Synthetic-board checks for vision/camera.py: no camera."""
import cv2
import numpy as np

from vision import camera

IMAGE_SIZE = (1280, 720)
K = np.array([[910.0, 0.0, 645.0], [0.0, 905.0, 355.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.08, -0.15, 0.001, -0.0005, 0.05])
BOARD = (9, 6)


def views(n=15, noise_px=0.2):
    """Board corners as the camera would see them from `n` tilted poses."""
    rng = np.random.default_rng(0)
    obj = camera.board_points(BOARD) * 0.025  # 25 mm squares
    out = []
    while len(out) < n:
        rvec = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.0), rng.uniform(0.35, 0.7)])
        px, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
        px = px.reshape(-1, 2)
        if (px < 0).any() or (px >= IMAGE_SIZE).any():
            continue
        out.append(px + rng.normal(0.0, noise_px, px.shape))
    return out


def test_calibration_recovers_the_intrinsics():
    cal, rms = camera.calibrate(views(), BOARD, IMAGE_SIZE)

    np.testing.assert_allclose(cal.camera_matrix, K, rtol=0.01, atol=0.5)
    assert rms < 0.5
    assert cal.image_size == IMAGE_SIZE


def test_yaml_round_trip(tmp_path):
    cal = camera.CameraCalibration(IMAGE_SIZE, K, DIST)
    path = tmp_path / "cam.yaml"

    camera.save(cal, path, "test_cam", note="rms 0.21 px\n15 views")
    back = camera.load(path)

    assert path.read_text().startswith("# rms 0.21 px\n# 15 views\n")
    assert back.image_size == IMAGE_SIZE
    np.testing.assert_allclose(back.camera_matrix, K)
    np.testing.assert_allclose(back.dist_coeffs, DIST)
