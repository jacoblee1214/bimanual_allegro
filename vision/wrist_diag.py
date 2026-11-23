"""wrist_diag.py — the /<side>/wrist_depth_diag message body, without ROS (NumPy-free, importable anywhere).

The fields of vision/hand_tracker_node_v4.py's message, unchanged, plus `gap_factor` only
when the fusion's output has that field (vision/depth_wrist_v4.py: held range over the
anchor range, null on frames that are not "hold"). Readers of the older messages see the
same keys, so nothing breaks for them.
"""
from __future__ import annotations

import json


def diag_json(fused, sigma_ray: float) -> str:
    pixels = getattr(fused, "pixels", None)
    body = {
        "source": fused.source,
        "scale": round(float(fused.scale), 4),
        "measured": None if fused.measured is None else round(float(fused.measured), 4),
        "pixels": None if pixels is None else int(pixels),
        "sigma_ray_mm": round(sigma_ray * 1000.0, 2),
    }
    if hasattr(fused, "gap_factor"):
        gap = fused.gap_factor
        body["gap_factor"] = None if gap is None else round(float(gap), 4)
    return json.dumps(body)
