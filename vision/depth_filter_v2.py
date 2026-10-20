"""
depth_filter_v2.py — RayKalman with the ray noise given per measurement.

v1 (depth_filter.py) has one `sigma_ray` per filter. Once the wrist distance comes from
the depth sensor on some frames and from PnP on others (vision/depth_wrist.py), the
same filter has to trust the ray direction a lot on the first kind (M5: 5 mm) and
little on the second (M4: 45 mm). tools/eval_depth_wrist.py did this by setting
`sigma_ray` before every call; here it is an argument, and the filter's own value is
left as it was.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vision.depth_filter import RayKalman


class RayKalmanV2(RayKalman):
    def __call__(self, z: np.ndarray, t: float, sigma_ray: Optional[float] = None) -> np.ndarray:
        """As RayKalman, with `sigma_ray` (metres) for this measurement only; None: the filter's own."""
        if sigma_ray is None:
            return super().__call__(z, t)
        own = self.sigma_ray
        self.sigma_ray = float(sigma_ray)
        try:
            return super().__call__(z, t)
        finally:
            self.sigma_ray = own
