#!/usr/bin/env python3
"""
analyze_hand_scale.py — Milestone M5 hand-scale study:
How much of monocular PnP wrist depth error is:
  (a) a constant per-operator, per-side scale factor,
  (b) a factor that varies with hand pose (palm normal angle to camera ray,
      finger curl, apparent hand size in pixels, image position),
  (c) unexplained frame-to-frame noise?

Uses pre-extracted detections from tools/extract_ego_detections.py
(data/results/m5_detections_<clip>.npz) and aligned depth maps from
tools/record_ego_depth.py (data/clips/<clip>/depth.npy, meta.json).
Does not re-run MediaPipe and does not import vision/depth_wrist.py.

Usage (with .venv-vision):
    .venv-vision/bin/python -B tools/analyze_hand_scale.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision import camera  # noqa: E402
from vision.wrist_pose import N_LANDMARKS, palm_frame  # noqa: E402

PALM_POLY = (0, 1, 5, 9, 13, 17)
MCPS = (5, 9, 13, 17)
FINGER_CHAINS = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
SIDES = ("left", "right")
PHASES = ("ready", "still", "depth", "free")
POSE_FEATURE_NAMES = (
    "cos_normal_ray",
    "finger_curl",
    "apparent_size_px",
    "img_u",
    "img_v",
)
OUT_DEFAULT = REPO / "data" / "results" / "m5_hand_scale.json"


def solve_hand_in_camera(
    world_landmarks: np.ndarray,
    image_landmarks: np.ndarray,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """Re-solve SQPnP on all 21 landmarks and return full (21, 3) camera-frame points and RMS reproj px."""
    world = np.asarray(world_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    image = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    if not (np.isfinite(world).all() and np.isfinite(image).all()):
        return None
    w, h = image_size
    pixels = image[:, :2] * (w, h)
    dist = np.zeros(5, dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(world, pixels, camera_matrix, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(world, rvec, tvec, camera_matrix, dist)
    reproj_px = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
    r_mat, _ = cv2.Rodrigues(rvec)
    in_camera = world @ r_mat.T + tvec.reshape(1, 3)
    return in_camera, reproj_px


def pnp_palm_depth(in_camera: np.ndarray) -> float:
    """PnP palm depth (m): mean camera optical Z of the palm polygon landmarks (0,1,5,9,13,17)."""
    pts = np.asarray(in_camera, dtype=np.float64).reshape(N_LANDMARKS, 3)
    return float(np.mean(pts[list(PALM_POLY), 2]))


def palm_polygon_mask(
    image_landmarks: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Binary uint8 mask (H, W) inside the 2D palm polygon (landmarks 0,1,5,9,13,17)."""
    w, h = image_size
    image = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    pts = np.round(image[list(PALM_POLY), :2] * (w, h)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def sensor_palm_depth(
    depth_frame: np.ndarray,
    image_landmarks: np.ndarray,
    depth_scale: float,
    image_size: tuple[int, int] = (640, 480),
    z_min: float = 0.15,
    z_max: float = 1.50,
    min_pixels: int = 15,
    min_valid_fraction: float = 0.20,
    max_mad_m: float = 0.06,
    mad_sigma: float = 3.0,
) -> dict[str, float | int | bool]:
    """Robust median of valid depth pixels inside the palm polygon (landmarks 0,1,5,9,13,17).

    Steps:
      1. Rasterize the 6-vertex palm polygon (0, 1, 5, 9, 13, 17).
      2. Keep depth pixels in [z_min, z_max] metres.
      3. Compute initial median and MAD, then trim outliers beyond max(20 mm, mad_sigma * 1.4826 * MAD)
         to reject boundary background leakage or stereo flying pixels.
      4. Flag `quality_ok` if polygon area >= min_pixels, inlier count >= min_pixels,
         inlier fraction >= min_valid_fraction, and initial MAD <= max_mad_m.
    """
    mask = palm_polygon_mask(image_landmarks, image_size)
    poly_area = int((mask == 1).sum())
    if poly_area == 0:
        return {
            "z_sensor_m": float("nan"),
            "z_raw_median_m": float("nan"),
            "poly_area_px": 0,
            "n_in_range_px": 0,
            "n_inlier_px": 0,
            "valid_fraction": 0.0,
            "poly_mad_m": float("nan"),
            "inlier_mad_m": float("nan"),
            "quality_ok": False,
        }

    vals = depth_frame[mask == 1].astype(np.float64) * float(depth_scale)
    valid = vals[(vals >= z_min) & (vals <= z_max)]
    n_valid = int(len(valid))
    if n_valid == 0:
        return {
            "z_sensor_m": float("nan"),
            "z_raw_median_m": float("nan"),
            "poly_area_px": poly_area,
            "n_in_range_px": 0,
            "n_inlier_px": 0,
            "valid_fraction": 0.0,
            "poly_mad_m": float("nan"),
            "inlier_mad_m": float("nan"),
            "quality_ok": False,
        }

    med0 = float(np.median(valid))
    mad0 = float(np.median(np.abs(valid - med0)))
    tol = max(0.020, mad_sigma * 1.4826 * mad0)
    inliers = valid[np.abs(valid - med0) <= tol]
    n_inliers = int(len(inliers))
    z_robust = float(np.median(inliers)) if n_inliers > 0 else med0
    mad_inliers = float(np.median(np.abs(inliers - z_robust))) if n_inliers > 0 else mad0
    valid_frac = float(n_inliers / poly_area)
    quality_ok = bool(
        poly_area >= min_pixels
        and n_inliers >= min_pixels
        and valid_frac >= min_valid_fraction
        and mad0 <= max_mad_m
    )
    return {
        "z_sensor_m": z_robust,
        "z_raw_median_m": med0,
        "poly_area_px": poly_area,
        "n_in_range_px": n_valid,
        "n_inlier_px": n_inliers,
        "valid_fraction": valid_frac,
        "poly_mad_m": mad0,
        "inlier_mad_m": mad_inliers,
        "quality_ok": quality_ok,
    }


def extract_pose_features(
    world_landmarks: np.ndarray,
    image_landmarks: np.ndarray,
    in_camera: np.ndarray,
    image_size: tuple[int, int],
    side: str,
) -> dict[str, float]:
    """Compute pose features per frame for scale regression and diagnostics.

    Features:
      - cos_normal_ray: dot(palm_normal, unit_camera_ray_to_palm). Positive when the back
        of the hand faces the camera, negative when the palm faces the camera.
      - normal_angle_deg: arccos(cos_normal_ray) in degrees.
      - finger_curl: mean over index/middle/ring/little of 1 - ||tip - mcp|| / sum(bone_lengths),
        dimensionless in [0, 1] (0 = straight fingers, higher = curled fist).
      - apparent_size_px: mean 2D pixel distance from wrist (0) to MCPs (5, 9, 13, 17).
      - img_u, img_v: normalized palm centroid relative to image centre (in [-0.5, 0.5]).
      - world_palm_scale_mm: MediaPipe world-landmark mean wrist-to-MCP distance (mm).
    """
    world = np.asarray(world_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    image = np.asarray(image_landmarks, dtype=np.float64).reshape(N_LANDMARKS, 3)
    cam = np.asarray(in_camera, dtype=np.float64).reshape(N_LANDMARKS, 3)
    w, h = image_size

    r_palm = palm_frame(cam, side)
    palm_normal = r_palm[:, 0]  # points out of the palm on both hands
    palm_center_cam = np.mean(cam[list(PALM_POLY)], axis=0)
    u_ray = palm_center_cam / np.linalg.norm(palm_center_cam)
    cos_normal_ray = float(np.clip(np.dot(palm_normal, u_ray), -1.0, 1.0))
    normal_angle_deg = float(np.degrees(np.arccos(cos_normal_ray)))

    curls = []
    for mcp, pip, dip, tip in FINGER_CHAINS:
        bone_len = (
            np.linalg.norm(world[pip] - world[mcp])
            + np.linalg.norm(world[dip] - world[pip])
            + np.linalg.norm(world[tip] - world[dip])
        )
        chord = np.linalg.norm(world[tip] - world[mcp])
        curls.append(1.0 - float(chord / max(bone_len, 1e-9)))
    finger_curl = float(np.mean(curls))

    pixels = image[:, :2] * (w, h)
    apparent_size_px = float(np.linalg.norm(pixels[list(MCPS)] - pixels[[0]], axis=1).mean())
    palm_uv = np.mean(image[list(PALM_POLY), :2], axis=0)
    img_u = float(palm_uv[0] - 0.5)
    img_v = float(palm_uv[1] - 0.5)
    world_palm_scale_mm = float(np.linalg.norm(world[list(MCPS)] - world[[0]], axis=1).mean() * 1000.0)

    return {
        "cos_normal_ray": cos_normal_ray,
        "normal_angle_deg": normal_angle_deg,
        "finger_curl": finger_curl,
        "apparent_size_px": apparent_size_px,
        "img_u": img_u,
        "img_v": img_v,
        "world_palm_scale_mm": world_palm_scale_mm,
    }


def fit_linear_pose_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    l2_reg: float = 1e-4,
) -> dict[str, np.ndarray | float]:
    """Fit standardized ridge-stabilized OLS linear model y_hat = b0 + x_std @ w."""
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    std_x_safe = np.where(std_x > 1e-9, std_x, 1.0)
    x_std = (x - mean_x) / std_x_safe
    # Zero out columns that are practically constant in train
    x_std[:, std_x <= 1e-9] = 0.0
    mean_y = float(np.mean(y))
    y_c = y - mean_y
    n_feat = x.shape[1]
    a_mat = x_std.T @ x_std + l2_reg * max(len(y), 1) * np.eye(n_feat)
    w_std = np.linalg.solve(a_mat, x_std.T @ y_c)
    return {
        "mean_x": mean_x,
        "std_x": std_x_safe,
        "active": (std_x > 1e-9),
        "intercept": mean_y,
        "w_std": w_std,
    }


def predict_linear_pose_model(model: dict, x_eval: np.ndarray) -> np.ndarray:
    x = np.asarray(x_eval, dtype=np.float64)
    x_std = (x - model["mean_x"]) / model["std_x"]
    x_std[:, ~model["active"]] = 0.0
    return float(model["intercept"]) + x_std @ model["w_std"]


def summarize_distribution(vals: np.ndarray) -> dict[str, float | int | None]:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(len(v))
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "cv": None,
            "median": None,
            "iqr": None,
            "mad": None,
            "p5": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    mean = float(np.mean(v))
    std = float(np.std(v))
    med = float(np.median(v))
    p5, p25, p75, p95 = [float(x) for x in np.percentile(v, [5, 25, 75, 95])]
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "cv": round(float(std / mean), 6) if abs(mean) > 1e-12 else None,
        "median": round(med, 6),
        "iqr": round(float(p75 - p25), 6),
        "mad": round(float(np.median(np.abs(v - med))), 6),
        "p5": round(p5, 6),
        "p25": round(p25, 6),
        "p75": round(p75, 6),
        "p95": round(p95, 6),
        "min": round(float(np.min(v)), 6),
        "max": round(float(np.max(v)), 6),
    }


def evaluate_predictions(
    s_true: np.ndarray,
    z_pnp: np.ndarray,
    z_sensor: np.ndarray,
    s_hat: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute variance explained and residual depth error in mm for predicted scale s_hat."""
    n = int(len(s_true))
    if n == 0:
        return {
            "n": 0,
            "r2_uncalibrated_scale": None,
            "r2_variance_within_set": None,
            "r2_uncalibrated_depth": None,
            "scale_rmse": None,
            "scale_median_abs": None,
            "scale_p95_abs": None,
            "depth_err_median_abs_mm": None,
            "depth_err_p95_abs_mm": None,
            "depth_err_rmse_mm": None,
            "depth_err_bias_mm": None,
        }

    s_err = s_hat - s_true
    sse_scale = float(np.sum(s_err ** 2))
    sst_uncal_scale = float(np.sum((s_true - 1.0) ** 2))
    sst_var_scale = float(np.sum((s_true - np.mean(s_true)) ** 2))

    z_hat = s_hat * z_pnp
    z_err_m = z_hat - z_sensor
    z_err_mm = z_err_m * 1000.0
    abs_err_mm = np.abs(z_err_mm)
    sse_depth = float(np.sum(z_err_m ** 2))
    sst_uncal_depth = float(np.sum((z_pnp - z_sensor) ** 2))

    return {
        "n": n,
        "r2_uncalibrated_scale": round(1.0 - sse_scale / sst_uncal_scale, 6) if sst_uncal_scale > 1e-12 else None,
        "r2_variance_within_set": round(1.0 - sse_scale / sst_var_scale, 6) if sst_var_scale > 1e-12 else None,
        "r2_uncalibrated_depth": round(1.0 - sse_depth / sst_uncal_depth, 6) if sst_uncal_depth > 1e-12 else None,
        "scale_rmse": round(float(np.sqrt(np.mean(s_err ** 2))), 6),
        "scale_median_abs": round(float(np.median(np.abs(s_err))), 6),
        "scale_p95_abs": round(float(np.percentile(np.abs(s_err), 95)), 6),
        "depth_err_median_abs_mm": round(float(np.median(abs_err_mm)), 3),
        "depth_err_p95_abs_mm": round(float(np.percentile(abs_err_mm, 95)), 3),
        "depth_err_rmse_mm": round(float(np.sqrt(np.mean(z_err_mm ** 2))), 3),
        "depth_err_bias_mm": round(float(np.mean(z_err_mm)), 3),
    }


def run_split_evaluation(
    s_true: np.ndarray,
    z_pnp: np.ndarray,
    z_sensor: np.ndarray,
    x_pose: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    min_train: int = 8,
    min_test: int = 5,
) -> dict:
    """Train constant and linear-pose models on train_mask and evaluate on both train and test."""
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    if n_train < min_train or n_test < min_test:
        return {
            "n_train": n_train,
            "n_test": n_test,
            "valid_split": False,
            "const_scale_mean": None,
            "const_scale_median": None,
            "linear_weights_std": None,
            "train": None,
            "test": None,
        }

    s_tr, z_pnp_tr, z_sens_tr, x_tr = s_true[train_mask], z_pnp[train_mask], z_sensor[train_mask], x_pose[train_mask]
    s_te, z_pnp_te, z_sens_te, x_te = s_true[test_mask], z_pnp[test_mask], z_sensor[test_mask], x_pose[test_mask]

    c_mean = float(np.mean(s_tr))
    c_med = float(np.median(s_tr))
    lin_mod = fit_linear_pose_model(x_tr, s_tr)

    s_hat_tr_lin = predict_linear_pose_model(lin_mod, x_tr)
    s_hat_te_lin = predict_linear_pose_model(lin_mod, x_te)

    train_eval = {
        "no_correction": evaluate_predictions(s_tr, z_pnp_tr, z_sens_tr, np.ones_like(s_tr)),
        "const_per_side": evaluate_predictions(s_tr, z_pnp_tr, z_sens_tr, np.full_like(s_tr, c_mean)),
        "const_median_per_side": evaluate_predictions(s_tr, z_pnp_tr, z_sens_tr, np.full_like(s_tr, c_med)),
        "linear_pose_per_side": evaluate_predictions(s_tr, z_pnp_tr, z_sens_tr, s_hat_tr_lin),
    }
    test_eval = {
        "no_correction": evaluate_predictions(s_te, z_pnp_te, z_sens_te, np.ones_like(s_te)),
        "const_per_side": evaluate_predictions(s_te, z_pnp_te, z_sens_te, np.full_like(s_te, c_mean)),
        "const_median_per_side": evaluate_predictions(s_te, z_pnp_te, z_sens_te, np.full_like(s_te, c_med)),
        "linear_pose_per_side": evaluate_predictions(s_te, z_pnp_te, z_sens_te, s_hat_te_lin),
    }

    return {
        "n_train": n_train,
        "n_test": n_test,
        "valid_split": True,
        "const_scale_mean": round(c_mean, 6),
        "const_scale_median": round(c_med, 6),
        "linear_weights_std": {
            name: round(float(w), 6) for name, w in zip(POSE_FEATURE_NAMES, lin_mod["w_std"])
        },
        "train": train_eval,
        "test": test_eval,
    }


def compute_still_noise_floor(
    frames: np.ndarray,
    z_sensor: np.ndarray,
    z_pnp: np.ndarray,
    inlier_mad_m: np.ndarray,
    still_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute the depth sensor's still-phase depth spread (noise floor) vs PnP on the same frames."""
    idx = np.flatnonzero(still_mask)
    n = int(len(idx))
    if n < 3:
        return {
            "n_still_frames": n,
            "sensor_median_m": None,
            "sensor_std_mm": None,
            "sensor_iqr_mm": None,
            "sensor_mad_mm": None,
            "sensor_p95_minus_p5_mm": None,
            "sensor_d1_median_abs_mm": None,
            "sensor_d1_p95_abs_mm": None,
            "sensor_d2_median_abs_mm": None,
            "sensor_d2_p95_abs_mm": None,
            "sensor_dev5_rms_mm": None,
            "sensor_dev5_median_abs_mm": None,
            "sensor_dev5_p95_abs_mm": None,
            "sensor_within_poly_mad_median_mm": None,
            "pnp_median_m": None,
            "pnp_std_mm": None,
            "pnp_iqr_mm": None,
            "pnp_mad_mm": None,
            "pnp_d1_median_abs_mm": None,
            "pnp_d2_median_abs_mm": None,
            "pnp_dev5_rms_mm": None,
        }

    f_st = frames[idx]
    zs = z_sensor[idx] * 1000.0  # mm
    zp = z_pnp[idx] * 1000.0     # mm
    mad_poly = inlier_mad_m[idx] * 1000.0

    # Consecutive first differences
    cons1 = np.diff(f_st) == 1
    d1_s = np.abs(np.diff(zs))[cons1]
    d1_p = np.abs(np.diff(zp))[cons1]

    # Consecutive triplets for second difference (d2)
    cons2 = (f_st[2:] - f_st[:-2]) == 2
    d2_s = np.abs(zs[2:] - 2.0 * zs[1:-1] + zs[:-2])[cons2]
    d2_p = np.abs(zp[2:] - 2.0 * zp[1:-1] + zp[:-2])[cons2]

    # 5-frame centred moving median deviation on consecutive stretches
    dev5_s, dev5_p = [], []
    for j in range(2, n - 2):
        if f_st[j + 2] - f_st[j - 2] == 4:
            dev5_s.append(zs[j] - float(np.median(zs[j - 2 : j + 3])))
            dev5_p.append(zp[j] - float(np.median(zp[j - 2 : j + 3])))
    dev5_s = np.asarray(dev5_s, dtype=np.float64)
    dev5_p = np.asarray(dev5_p, dtype=np.float64)

    s_p5, s_p25, s_p75, s_p95 = [float(x) for x in np.percentile(zs, [5, 25, 75, 95])]
    p_p25, p_p75 = [float(x) for x in np.percentile(zp, [25, 75])]

    return {
        "n_still_frames": n,
        "n_consecutive_pairs": int(len(d1_s)),
        "n_consecutive_triplets": int(len(d2_s)),
        "n_consecutive_5win": int(len(dev5_s)),
        "sensor_median_m": round(float(np.median(zs) / 1000.0), 6),
        "sensor_std_mm": round(float(np.std(zs)), 3),
        "sensor_iqr_mm": round(float(s_p75 - s_p25), 3),
        "sensor_mad_mm": round(float(np.median(np.abs(zs - np.median(zs)))), 3),
        "sensor_p95_minus_p5_mm": round(float(s_p95 - s_p5), 3),
        "sensor_d1_median_abs_mm": round(float(np.median(d1_s)), 3) if len(d1_s) else None,
        "sensor_d1_p95_abs_mm": round(float(np.percentile(d1_s, 95)), 3) if len(d1_s) else None,
        "sensor_d2_median_abs_mm": round(float(np.median(d2_s)), 3) if len(d2_s) else None,
        "sensor_d2_p95_abs_mm": round(float(np.percentile(d2_s, 95)), 3) if len(d2_s) else None,
        "sensor_dev5_rms_mm": round(float(np.sqrt(np.mean(dev5_s ** 2))), 3) if len(dev5_s) else None,
        "sensor_dev5_median_abs_mm": round(float(np.median(np.abs(dev5_s))), 3) if len(dev5_s) else None,
        "sensor_dev5_p95_abs_mm": round(float(np.percentile(np.abs(dev5_s), 95)), 3) if len(dev5_s) else None,
        "sensor_within_poly_mad_median_mm": round(float(np.median(mad_poly)), 3),
        "pnp_median_m": round(float(np.median(zp) / 1000.0), 6),
        "pnp_std_mm": round(float(np.std(zp)), 3),
        "pnp_iqr_mm": round(float(p_p75 - p_p25), 3),
        "pnp_mad_mm": round(float(np.median(np.abs(zp - np.median(zp)))), 3),
        "pnp_d1_median_abs_mm": round(float(np.median(d1_p)), 3) if len(d1_p) else None,
        "pnp_d2_median_abs_mm": round(float(np.median(d2_p)), 3) if len(d2_p) else None,
        "pnp_dev5_rms_mm": round(float(np.sqrt(np.mean(dev5_p ** 2))), 3) if len(dev5_p) else None,
    }


def analyze_side_records(
    records: list[dict],
) -> dict:
    """Run all statistical summaries, variance decompositions, and split evaluations for one side."""
    n_found = len(records)
    valid_recs = [r for r in records if r["quality_ok"] and np.isfinite(r["s_true"])]
    n_valid = len(valid_recs)

    # Also summarize all frames that had any in-range depth pixel (even if quality_ok=False)
    any_recs = [r for r in records if np.isfinite(r["s_true_raw"])]

    if n_valid == 0:
        return {
            "n_found": n_found,
            "n_any_depth": len(any_recs),
            "n_valid": 0,
            "s_true_by_phase": {},
            "still_noise_floor": None,
            "feature_correlations": {},
            "decomposition_in_sample": None,
            "splits": {},
        }

    frames = np.array([r["frame"] for r in valid_recs], dtype=np.int64)
    phases = np.array([r["phase"] for r in valid_recs])
    s_true = np.array([r["s_true"] for r in valid_recs], dtype=np.float64)
    z_pnp = np.array([r["z_pnp_m"] for r in valid_recs], dtype=np.float64)
    z_sensor = np.array([r["z_sensor_m"] for r in valid_recs], dtype=np.float64)
    inlier_mad = np.array([r["inlier_mad_m"] for r in valid_recs], dtype=np.float64)
    x_pose = np.array(
        [[r["features"][k] for k in POSE_FEATURE_NAMES] for r in valid_recs],
        dtype=np.float64,
    )

    # 1. Distribution of s_true per phase
    s_true_by_phase = {
        "all": {
            "s_true": summarize_distribution(s_true),
            "z_sensor_m": summarize_distribution(z_sensor),
            "z_pnp_m": summarize_distribution(z_pnp),
            "raw_depth_err_mm": summarize_distribution((z_pnp - z_sensor) * 1000.0),
        }
    }
    for ph in PHASES:
        m = phases == ph
        s_true_by_phase[ph] = {
            "n_found_phase": int(sum(1 for r in records if r["phase"] == ph)),
            "s_true": summarize_distribution(s_true[m]),
            "z_sensor_m": summarize_distribution(z_sensor[m]),
            "z_pnp_m": summarize_distribution(z_pnp[m]),
            "raw_depth_err_mm": summarize_distribution((z_pnp[m] - z_sensor[m]) * 1000.0),
        }

    # 2. Still-phase sensor depth noise floor
    still_noise_floor = compute_still_noise_floor(
        frames=frames,
        z_sensor=z_sensor,
        z_pnp=z_pnp,
        inlier_mad_m=inlier_mad,
        still_mask=(phases == "still"),
    )

    # 3. Single-feature correlations with s_true
    feature_correlations = {}
    all_feat_names = list(POSE_FEATURE_NAMES) + ["normal_angle_deg", "world_palm_scale_mm"]
    for fname in all_feat_names:
        fvals = np.array([r["features"][fname] for r in valid_recs], dtype=np.float64)
        if len(fvals) >= 5 and np.std(fvals) > 1e-9 and np.std(s_true) > 1e-9:
            r_val = float(np.corrcoef(fvals, s_true)[0, 1])
            slope = float(np.polyfit(fvals, s_true, 1)[0])
            feature_correlations[fname] = {
                "r": round(r_val, 6),
                "r2": round(r_val ** 2, 6),
                "slope": round(slope, 6),
                "feature_median": round(float(np.median(fvals)), 4),
                "feature_iqr": round(float(np.percentile(fvals, 75) - np.percentile(fvals, 25)), 4),
            }
        else:
            feature_correlations[fname] = {"r": None, "r2": None, "slope": None}

    # 4. Orthogonal 3-way decomposition of total uncalibrated squared scale error E[(s_true - 1)^2]
    #    E[(s - 1)^2] = (mean(s) - 1)^2 [constant per-side] + Var(s_hat_pose) [pose] + E[(s - s_hat_pose)^2] [noise]
    all_mask = np.ones(n_valid, dtype=bool)
    in_sample_split = run_split_evaluation(s_true, z_pnp, z_sensor, x_pose, all_mask, all_mask)
    decomposition = None
    if in_sample_split["valid_split"]:
        lin_mod_all = fit_linear_pose_model(x_pose, s_true)
        s_hat_pose = predict_linear_pose_model(lin_mod_all, x_pose)
        mse_total = float(np.mean((s_true - 1.0) ** 2))
        mse_const = float((np.mean(s_true) - 1.0) ** 2)
        var_within = float(np.var(s_true))
        mse_resid = float(np.mean((s_true - s_hat_pose) ** 2))
        var_pose = max(0.0, var_within - mse_resid)

        # Also in depth space (mm^2): E[(z_pnp - z_sensor)^2]
        d_mse_total = float(np.mean(((z_pnp - z_sensor) * 1000.0) ** 2))
        d_mse_after_const = float(np.mean(((np.mean(s_true) * z_pnp - z_sensor) * 1000.0) ** 2))
        d_mse_after_pose = float(np.mean(((s_hat_pose * z_pnp - z_sensor) * 1000.0) ** 2))
        d_const_explained = max(0.0, d_mse_total - d_mse_after_const)
        d_pose_explained = max(0.0, d_mse_after_const - d_mse_after_pose)

        decomposition = {
            "scale_mse_total_vs_1": round(mse_total, 6),
            "scale_const_bias_sq": round(mse_const, 6),
            "scale_within_side_var": round(var_within, 6),
            "scale_pose_var_explained": round(var_pose, 6),
            "scale_unexplained_noise_mse": round(mse_resid, 6),
            "frac_of_uncalibrated_scale_mse": {
                "a_constant_per_side": round(mse_const / mse_total, 6) if mse_total > 1e-12 else None,
                "b_pose_linear": round(var_pose / mse_total, 6) if mse_total > 1e-12 else None,
                "c_unexplained_noise": round(mse_resid / mse_total, 6) if mse_total > 1e-12 else None,
            },
            "frac_of_within_side_scale_var": {
                "b_pose_linear": round(var_pose / var_within, 6) if var_within > 1e-12 else None,
                "c_unexplained_noise": round(mse_resid / var_within, 6) if var_within > 1e-12 else None,
            },
            "frac_of_uncalibrated_depth_mse": {
                "a_constant_per_side": round(d_const_explained / d_mse_total, 6) if d_mse_total > 1e-12 else None,
                "b_pose_linear": round(d_pose_explained / d_mse_total, 6) if d_mse_total > 1e-12 else None,
                "c_unexplained_noise": round(d_mse_after_pose / d_mse_total, 6) if d_mse_total > 1e-12 else None,
            },
        }

    # 5. Splits: time_split (first half -> second half), time_split_reverse (second half -> first half),
    #            still_to_free (still -> free), still_to_depth (still -> depth)
    half = n_valid // 2
    time_train = np.arange(n_valid) < half
    time_test = np.arange(n_valid) >= half

    still_train = phases == "still"
    free_test = phases == "free"
    depth_test = phases == "depth"

    splits = {
        "in_sample": in_sample_split,
        "time_split": run_split_evaluation(s_true, z_pnp, z_sensor, x_pose, time_train, time_test),
        "time_split_reverse": run_split_evaluation(s_true, z_pnp, z_sensor, x_pose, time_test, time_train),
        "still_to_free": run_split_evaluation(s_true, z_pnp, z_sensor, x_pose, still_train, free_test),
        "still_to_depth": run_split_evaluation(s_true, z_pnp, z_sensor, x_pose, still_train, depth_test),
    }

    return {
        "n_found": n_found,
        "n_any_depth": len(any_recs),
        "n_valid": n_valid,
        "valid_rate_of_found": round(float(n_valid / max(1, n_found)), 4),
        "valid_fraction_found_summary": summarize_distribution([r["valid_fraction"] for r in records]),
        "valid_fraction_valid_summary": summarize_distribution([r["valid_fraction"] for r in valid_recs]),
        "any_depth_s_true_summary": summarize_distribution([r["s_true_raw"] for r in any_recs]),
        "s_true_by_phase": s_true_by_phase,
        "still_noise_floor": still_noise_floor,
        "feature_correlations": feature_correlations,
        "decomposition_in_sample": decomposition,
        "splits": splits,
    }


def analyze_pooled_sides(side_records: dict[str, list[dict]]) -> dict:
    """Pooled variance decomposition across both sides (left + right)."""
    valid_by_side = {
        s: [r for r in side_records[s] if r["quality_ok"] and np.isfinite(r["s_true"])]
        for s in SIDES
    }
    if any(len(valid_by_side[s]) < 8 for s in SIDES):
        return {"valid": False}

    # Combine for in-sample and time-split pooled metrics
    s_all, zp_all, zs_all = [], [], []
    s_hat_const_side, s_hat_lin_side = [], []

    # Time split pooled arrays
    s_te_all, zp_te_all, zs_te_all = [], [], []
    s_te_const_side, s_te_lin_side = [], []

    # Still-to-free pooled arrays
    s_fr_all, zp_fr_all, zs_fr_all = [], [], []
    s_fr_const_side, s_fr_lin_side = [], []
    has_still_free = True

    for s in SIDES:
        recs = valid_by_side[s]
        n = len(recs)
        s_vec = np.array([r["s_true"] for r in recs], dtype=np.float64)
        zp_vec = np.array([r["z_pnp_m"] for r in recs], dtype=np.float64)
        zs_vec = np.array([r["z_sensor_m"] for r in recs], dtype=np.float64)
        ph_vec = np.array([r["phase"] for r in recs])
        x_mat = np.array([[r["features"][k] for k in POSE_FEATURE_NAMES] for r in recs], dtype=np.float64)

        # In-sample per side
        c_s = float(np.mean(s_vec))
        lin_s = fit_linear_pose_model(x_mat, s_vec)
        s_all.append(s_vec)
        zp_all.append(zp_vec)
        zs_all.append(zs_vec)
        s_hat_const_side.append(np.full_like(s_vec, c_s))
        s_hat_lin_side.append(predict_linear_pose_model(lin_s, x_mat))

        # Time split per side
        half = n // 2
        tr = np.arange(n) < half
        te = np.arange(n) >= half
        c_tr = float(np.mean(s_vec[tr]))
        lin_tr = fit_linear_pose_model(x_mat[tr], s_vec[tr])
        s_te_all.append(s_vec[te])
        zp_te_all.append(zp_vec[te])
        zs_te_all.append(zs_vec[te])
        s_te_const_side.append(np.full_like(s_vec[te], c_tr))
        s_te_lin_side.append(predict_linear_pose_model(lin_tr, x_mat[te]))

        # Still-to-free per side
        st_m = ph_vec == "still"
        fr_m = ph_vec == "free"
        if st_m.sum() >= 8 and fr_m.sum() >= 5:
            c_st = float(np.mean(s_vec[st_m]))
            lin_st = fit_linear_pose_model(x_mat[st_m], s_vec[st_m])
            s_fr_all.append(s_vec[fr_m])
            zp_fr_all.append(zp_vec[fr_m])
            zs_fr_all.append(zs_vec[fr_m])
            s_fr_const_side.append(np.full_like(s_vec[fr_m], c_st))
            s_fr_lin_side.append(predict_linear_pose_model(lin_st, x_mat[fr_m]))
        else:
            has_still_free = False

    s_all = np.concatenate(s_all)
    zp_all = np.concatenate(zp_all)
    zs_all = np.concatenate(zs_all)
    s_hat_const_side = np.concatenate(s_hat_const_side)
    s_hat_lin_side = np.concatenate(s_hat_lin_side)

    s_te_all = np.concatenate(s_te_all)
    zp_te_all = np.concatenate(zp_te_all)
    zs_te_all = np.concatenate(zs_te_all)
    s_te_const_side = np.concatenate(s_te_const_side)
    s_te_lin_side = np.concatenate(s_te_lin_side)

    out = {
        "valid": True,
        "in_sample": {
            "no_correction": evaluate_predictions(s_all, zp_all, zs_all, np.ones_like(s_all)),
            "const_per_side": evaluate_predictions(s_all, zp_all, zs_all, s_hat_const_side),
            "linear_pose_per_side": evaluate_predictions(s_all, zp_all, zs_all, s_hat_lin_side),
        },
        "time_split_test": {
            "no_correction": evaluate_predictions(s_te_all, zp_te_all, zs_te_all, np.ones_like(s_te_all)),
            "const_per_side": evaluate_predictions(s_te_all, zp_te_all, zs_te_all, s_te_const_side),
            "linear_pose_per_side": evaluate_predictions(s_te_all, zp_te_all, zs_te_all, s_te_lin_side),
        },
    }
    if has_still_free:
        s_fr_all = np.concatenate(s_fr_all)
        zp_fr_all = np.concatenate(zp_fr_all)
        zs_fr_all = np.concatenate(zs_fr_all)
        s_fr_const_side = np.concatenate(s_fr_const_side)
        s_fr_lin_side = np.concatenate(s_fr_lin_side)
        out["still_to_free_test"] = {
            "no_correction": evaluate_predictions(s_fr_all, zp_fr_all, zs_fr_all, np.ones_like(s_fr_all)),
            "const_per_side": evaluate_predictions(s_fr_all, zp_fr_all, zs_fr_all, s_fr_const_side),
            "linear_pose_per_side": evaluate_predictions(s_fr_all, zp_fr_all, zs_fr_all, s_fr_lin_side),
        }
    else:
        out["still_to_free_test"] = None
    return out


def evaluate_cross_session_pair(
    train_side_records: dict[str, list[dict]],
    test_side_records: dict[str, list[dict]],
    min_train: int = 8,
    min_test: int = 5,
) -> dict:
    """Fit per-side constant (mean and median) and linear pose model on train_side_records (quality_ok)
    and evaluate on test_side_records (quality_ok)."""
    sides_out = {}
    s_te_all, zp_te_all, zs_te_all = [], [], []
    s_hat_mean_all, s_hat_med_all, s_hat_lin_all = [], [], []
    s_hat_oracle_mean_all, s_hat_oracle_med_all = [], []
    both_valid = True

    for s in SIDES:
        tr_recs = [r for r in train_side_records.get(s, []) if r["quality_ok"] and np.isfinite(r["s_true"])]
        te_recs = [r for r in test_side_records.get(s, []) if r["quality_ok"] and np.isfinite(r["s_true"])]
        n_tr = len(tr_recs)
        n_te = len(te_recs)
        if n_tr < min_train or n_te < min_test:
            both_valid = False
            sides_out[s] = {
                "n_train": n_tr,
                "n_test": n_te,
                "valid_split": False,
                "train_const_mean": None,
                "train_const_median": None,
                "test_const_mean": None,
                "test_const_median": None,
                "scale_shift_mean": None,
                "scale_shift_median": None,
                "test": None,
            }
            continue

        s_tr = np.array([r["s_true"] for r in tr_recs], dtype=np.float64)
        x_tr = np.array([[r["features"][k] for k in POSE_FEATURE_NAMES] for r in tr_recs], dtype=np.float64)

        s_te = np.array([r["s_true"] for r in te_recs], dtype=np.float64)
        zp_te = np.array([r["z_pnp_m"] for r in te_recs], dtype=np.float64)
        zs_te = np.array([r["z_sensor_m"] for r in te_recs], dtype=np.float64)
        x_te = np.array([[r["features"][k] for k in POSE_FEATURE_NAMES] for r in te_recs], dtype=np.float64)

        c_tr_mean = float(np.mean(s_tr))
        c_tr_med = float(np.median(s_tr))
        c_te_mean = float(np.mean(s_te))
        c_te_med = float(np.median(s_te))
        lin_mod = fit_linear_pose_model(x_tr, s_tr)

        s_hat_mean = np.full_like(s_te, c_tr_mean)
        s_hat_med = np.full_like(s_te, c_tr_med)
        s_hat_lin = predict_linear_pose_model(lin_mod, x_te)
        s_hat_oracle_mean = np.full_like(s_te, c_te_mean)
        s_hat_oracle_med = np.full_like(s_te, c_te_med)

        sides_out[s] = {
            "n_train": n_tr,
            "n_test": n_te,
            "valid_split": True,
            "train_const_mean": round(c_tr_mean, 6),
            "train_const_median": round(c_tr_med, 6),
            "test_const_mean": round(c_te_mean, 6),
            "test_const_median": round(c_te_med, 6),
            "scale_shift_mean": round(c_tr_mean - c_te_mean, 6),
            "scale_shift_median": round(c_tr_med - c_te_med, 6),
            "test": {
                "no_correction": evaluate_predictions(s_te, zp_te, zs_te, np.ones_like(s_te)),
                "const_per_side": evaluate_predictions(s_te, zp_te, zs_te, s_hat_mean),
                "const_median_per_side": evaluate_predictions(s_te, zp_te, zs_te, s_hat_med),
                "linear_pose_per_side": evaluate_predictions(s_te, zp_te, zs_te, s_hat_lin),
                "oracle_test_const_mean": evaluate_predictions(s_te, zp_te, zs_te, s_hat_oracle_mean),
                "oracle_test_const_median": evaluate_predictions(s_te, zp_te, zs_te, s_hat_oracle_med),
            },
        }

        s_te_all.append(s_te)
        zp_te_all.append(zp_te)
        zs_te_all.append(zs_te)
        s_hat_mean_all.append(s_hat_mean)
        s_hat_med_all.append(s_hat_med)
        s_hat_lin_all.append(s_hat_lin)
        s_hat_oracle_mean_all.append(s_hat_oracle_mean)
        s_hat_oracle_med_all.append(s_hat_oracle_med)

    pooled = {"valid": False}
    if both_valid and len(s_te_all) == len(SIDES):
        s_cat = np.concatenate(s_te_all)
        zp_cat = np.concatenate(zp_te_all)
        zs_cat = np.concatenate(zs_te_all)
        pooled = {
            "valid": True,
            "test": {
                "no_correction": evaluate_predictions(s_cat, zp_cat, zs_cat, np.ones_like(s_cat)),
                "const_per_side": evaluate_predictions(s_cat, zp_cat, zs_cat, np.concatenate(s_hat_mean_all)),
                "const_median_per_side": evaluate_predictions(s_cat, zp_cat, zs_cat, np.concatenate(s_hat_med_all)),
                "linear_pose_per_side": evaluate_predictions(s_cat, zp_cat, zs_cat, np.concatenate(s_hat_lin_all)),
                "oracle_test_const_mean": evaluate_predictions(
                    s_cat, zp_cat, zs_cat, np.concatenate(s_hat_oracle_mean_all)
                ),
                "oracle_test_const_median": evaluate_predictions(
                    s_cat, zp_cat, zs_cat, np.concatenate(s_hat_oracle_med_all)
                ),
            },
        }

    return {
        "sides": sides_out,
        "pooled_both_sides": pooled,
    }


def run_cross_session_evaluations(
    records_by_clip: dict[str, dict[str, list[dict]]],
) -> dict[str, dict]:
    """Build within-clip half splits and all cross-clip train->test evaluations on quality-gated frames."""
    out: dict[str, dict] = {}
    clip_names = list(records_by_clip.keys())

    # 1. Within-clip half splits (half1 -> half2 and half2 -> half1) on quality-gated records
    for cname in clip_names:
        s_recs = records_by_clip[cname]
        half1_recs: dict[str, list[dict]] = {}
        half2_recs: dict[str, list[dict]] = {}
        for s in SIDES:
            v = [r for r in s_recs[s] if r["quality_ok"] and np.isfinite(r["s_true"])]
            h = len(v) // 2
            half1_recs[s] = v[:h]
            half2_recs[s] = v[h:]
        out[f"{cname}_half1_to_half2"] = {
            "train_clip": f"{cname}_half1",
            "test_clip": f"{cname}_half2",
            "mode": "within_clip_halves",
            **evaluate_cross_session_pair(half1_recs, half2_recs),
        }
        out[f"{cname}_half2_to_half1"] = {
            "train_clip": f"{cname}_half2",
            "test_clip": f"{cname}_half1",
            "mode": "within_clip_halves",
            **evaluate_cross_session_pair(half2_recs, half1_recs),
        }

    # 2. Cross-clip pairs (train_clip -> test_clip)
    for tr_clip in clip_names:
        for te_clip in clip_names:
            if tr_clip == te_clip:
                continue
            key = f"{tr_clip}_to_{te_clip}"
            out[key] = {
                "train_clip": tr_clip,
                "test_clip": te_clip,
                "mode": "cross_clip",
                **evaluate_cross_session_pair(records_by_clip[tr_clip], records_by_clip[te_clip]),
            }

    return out


def analyze_clip(
    clip_name: str,
    cal: camera.CameraCalibration,
    z_min: float = 0.15,
    z_max: float = 1.50,
    min_pixels: int = 15,
    min_valid_fraction: float = 0.20,
    max_mad_m: float = 0.06,
) -> Optional[dict]:
    clip_dir = REPO / "data" / "clips" / clip_name
    det_path = REPO / "data" / "results" / f"m5_detections_{clip_name}.npz"
    if not (clip_dir / "depth.npy").exists() or not (clip_dir / "meta.json").exists() or not det_path.exists():
        return None

    meta = json.loads((clip_dir / "meta.json").read_text())
    depth_scale = float(meta["depth_scale"])
    ds_raw = meta.get("depth_stream", {"width": 640, "height": 480, "fx": 392.0})
    depth_fx = float(ds_raw["fx"])
    depth_stream_info = {
        "width": int(ds_raw["width"]),
        "height": int(ds_raw["height"]),
        "fx": round(depth_fx, 6),
        "baseline_m": 0.095,
        "max_disparity_px": 126,
        "min_z_m": round(float(depth_fx * 0.095 / 126.0), 4),
    }
    det = np.load(det_path)
    depth_all = np.load(clip_dir / "depth.npy", mmap_mode="r")

    n_frames = int(len(det["t"]))
    phases = [str(p) for p in det["phase"]]
    phase_counts = {ph: int(sum(1 for p in phases if p == ph)) for ph in PHASES}

    side_records: dict[str, list[dict]] = {s: [] for s in SIDES}
    max_wrist_re_solve_diff_m: dict[str, float] = {s: 0.0 for s in SIDES}

    for s in SIDES:
        found_arr = det[f"{s}_found"]
        world_arr = det[f"{s}_world"]
        image_arr = det[f"{s}_image"]
        p_stored_arr = det[f"{s}_p"]
        score_arr = det[f"{s}_score"]

        for i in np.flatnonzero(found_arr):
            i = int(i)
            solved = solve_hand_in_camera(
                world_landmarks=world_arr[i],
                image_landmarks=image_arr[i],
                image_size=cal.image_size,
                camera_matrix=cal.camera_matrix,
                dist_coeffs=cal.dist_coeffs,
            )
            if solved is None:
                continue
            in_camera, reproj_px = solved
            if np.isfinite(p_stored_arr[i]).all():
                diff_m = float(np.linalg.norm(in_camera[0] - p_stored_arr[i]))
                max_wrist_re_solve_diff_m[s] = max(max_wrist_re_solve_diff_m[s], diff_m)

            z_pnp = pnp_palm_depth(in_camera)
            z_pnp_med = float(np.median(in_camera[list(PALM_POLY), 2]))
            z_wrist_pnp = float(in_camera[0, 2])
            if z_pnp <= 0.05:
                continue

            d_info = sensor_palm_depth(
                depth_frame=depth_all[i],
                image_landmarks=image_arr[i],
                depth_scale=depth_scale,
                image_size=cal.image_size,
                z_min=z_min,
                z_max=z_max,
                min_pixels=min_pixels,
                min_valid_fraction=min_valid_fraction,
                max_mad_m=max_mad_m,
            )
            # Also record widest-range (up to 1.5 m) sensor median for diagnostic comparison
            d_wide = sensor_palm_depth(
                depth_frame=depth_all[i],
                image_landmarks=image_arr[i],
                depth_scale=depth_scale,
                image_size=cal.image_size,
                z_min=0.15,
                z_max=1.50,
                min_pixels=5,
                min_valid_fraction=0.0,
                max_mad_m=1.0,
            )

            z_sens = float(d_info["z_sensor_m"])
            s_true = float(z_sens / z_pnp) if np.isfinite(z_sens) else float("nan")
            z_sens_raw = float(d_wide["z_sensor_m"])
            s_true_raw = float(z_sens_raw / z_pnp) if np.isfinite(z_sens_raw) else float("nan")

            feats = extract_pose_features(
                world_landmarks=world_arr[i],
                image_landmarks=image_arr[i],
                in_camera=in_camera,
                image_size=cal.image_size,
                side=s,
            )

            side_records[s].append(
                {
                    "frame": i,
                    "t": float(det["t"][i]),
                    "phase": phases[i],
                    "score": float(score_arr[i]),
                    "reproj_px": reproj_px,
                    "z_pnp_m": z_pnp,
                    "z_pnp_median_m": z_pnp_med,
                    "z_wrist_pnp_m": z_wrist_pnp,
                    "z_sensor_m": z_sens,
                    "s_true": s_true,
                    "z_sensor_raw_m": z_sens_raw,
                    "s_true_raw": s_true_raw,
                    "poly_area_px": int(d_info["poly_area_px"]),
                    "n_inlier_px": int(d_info["n_inlier_px"]),
                    "valid_fraction": float(d_info["valid_fraction"]),
                    "poly_mad_m": float(d_info["poly_mad_m"]),
                    "inlier_mad_m": float(d_info["inlier_mad_m"]),
                    "quality_ok": bool(d_info["quality_ok"]),
                    "features": feats,
                }
            )

    sides_summary = {s: analyze_side_records(side_records[s]) for s in SIDES}
    pooled_summary = analyze_pooled_sides(side_records)

    return {
        "clip": clip_name,
        "n_frames": n_frames,
        "phase_frame_counts": phase_counts,
        "depth_scale": depth_scale,
        "depth_stream": depth_stream_info,
        "quality_gate": {
            "z_min_m": z_min,
            "z_max_m": z_max,
            "min_pixels": min_pixels,
            "min_valid_fraction": min_valid_fraction,
            "max_mad_m": max_mad_m,
        },
        "max_wrist_re_solve_diff_m": max_wrist_re_solve_diff_m,
        "sides": sides_summary,
        "pooled_both_sides": pooled_summary,
        "_side_records": side_records,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--clips",
        nargs="+",
        default=["smoke", "ego_depth", "ego_depth_work", "ego_depth_424"],
        help="clip names under data/clips/",
    )
    p.add_argument("--z-min", type=float, default=0.15, help="minimum valid sensor depth in metres (default 0.15)")
    p.add_argument("--z-max", type=float, default=1.50, help="maximum valid sensor depth in metres (default 1.50)")
    p.add_argument("--min-valid-frac", type=float, default=0.20, help="minimum inlier fraction in palm polygon")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    cal = camera.load(REPO / "config" / "ego_camera_info.yaml")
    clips_out = {}
    clips_workspace_out = {}
    clips_ungated_out = {}
    records_primary: dict[str, dict[str, list[dict]]] = {}
    records_workspace: dict[str, dict[str, list[dict]]] = {}

    for name in args.clips:
        res = analyze_clip(
            clip_name=name,
            cal=cal,
            z_min=args.z_min,
            z_max=args.z_max,
            min_valid_fraction=args.min_valid_frac,
        )
        if res is None:
            print(f"[!] clip {name!r} not found or incomplete, skipped")
            continue
        records_primary[name] = res.pop("_side_records")
        clips_out[name] = res

        # Workspace-only gate (z_max=1.00 m, excluding distant background detections >1.0 m)
        res_ws = analyze_clip(
            clip_name=name,
            cal=cal,
            z_min=args.z_min,
            z_max=1.00,
            min_pixels=15,
            min_valid_fraction=args.min_valid_frac,
            max_mad_m=0.06,
        )
        if res_ws is not None:
            records_workspace[name] = res_ws.pop("_side_records")
            clips_workspace_out[name] = res_ws

        # Ungated (all frames with >= 5 depth pixels in [0.15, 1.50] m, no hole/MAD rejection)
        res_ug = analyze_clip(
            clip_name=name,
            cal=cal,
            z_min=args.z_min,
            z_max=1.50,
            min_pixels=5,
            min_valid_fraction=0.0,
            max_mad_m=1.00,
        )
        if res_ug is not None:
            res_ug.pop("_side_records", None)
            clips_ungated_out[name] = res_ug

        for s in SIDES:
            sr = res["sides"][s]
            st_all = sr["s_true_by_phase"].get("all", {}).get("s_true", {})
            med_s = st_all.get("median")
            iqr_s = st_all.get("iqr")
            print(
                f"{name:15s} {s:5s}: found={sr['n_found']:3d}, valid={sr['n_valid']:3d}, "
                f"s_true med={med_s} (iqr={iqr_s})"
            )

    cross_session_out = run_cross_session_evaluations(records_primary)
    cross_session_ws_out = run_cross_session_evaluations(records_workspace)

    doc = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "tool": "tools/analyze_hand_scale.py",
            "camera_info": "config/ego_camera_info.yaml",
            "palm_polygon_landmarks": list(PALM_POLY),
            "pose_features": list(POSE_FEATURE_NAMES),
            "primary_quality_gate": {
                "z_min_m": args.z_min,
                "z_max_m": args.z_max,
                "min_pixels": 15,
                "min_valid_fraction": args.min_valid_frac,
                "max_mad_m": 0.06,
            },
            "workspace_quality_gate": {
                "z_min_m": args.z_min,
                "z_max_m": 1.00,
                "min_pixels": 15,
                "min_valid_fraction": args.min_valid_frac,
                "max_mad_m": 0.06,
            },
            "ungated_quality_gate": {
                "z_min_m": args.z_min,
                "z_max_m": 1.50,
                "min_pixels": 5,
                "min_valid_fraction": 0.0,
                "max_mad_m": 1.00,
            },
        },
        "clips": clips_out,
        "clips_workspace_zmax100": clips_workspace_out,
        "clips_ungated_zmax150": clips_ungated_out,
        "cross_session": cross_session_out,
        "cross_session_workspace_zmax100": cross_session_ws_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
