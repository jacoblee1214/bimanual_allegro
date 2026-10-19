"""
ik_v2.py — velocity-level (differential) 6D inverse kinematics for the 7-DOF xArm7.

Unlike `arm/ik.py`, which iterates damped least-squares to convergence every frame and
can jump across kinematic branches near singularities, `DiffIK.step` performs a single
linearized velocity step per control period:

1. 6D task twist in `LOCAL_WORLD_ALIGNED`:
       v = gain * [p_target - p ; rot_weight * log3(R_target R^T)]
2. Singularity-robust adaptive damping (Nakamura & Hanafusa, 1986, "Inverse Kinematic
   Solutions With Singularity Robustness for Robot Manipulator Control", ASME J. Dyn.
   Syst. Meas. Control 108(3):163-171; Chiaverini, Siciliano & Egeland, 1994, "Review
   of the Damped Least-Squares Inverse Kinematics with Experiments on an Industrial
   Robot Manipulator", IEEE Trans. Control Syst. Technol. 2(2):123-134):
       lambda^2 = damping^2 * (1 - (sigma_min / sigma_threshold)^2)  if sigma_min < sigma_threshold
                = 0                                                   otherwise
3. Redundancy resolution via null-space posture pull toward `q_rest`:
       dq = J^# v + (I - J^# J) (null_gain * (q_rest - q))
4. Uniform scaling of the entire `dq` vector so `max_i |dq_i| <= max_joint_speed_i * dt`
   (preserving the joint-space direction), followed by clamping `limit_margin` inside the
   URDF position limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "data" / "models" / "xarm7.urdf"

#: Default rest posture (rad) with the EEF at (0.3188, 0.0, 0.3933) m in `link_base`
#: and tool z-axis [-0.100, 0.0, -0.995] pointing downward (tilted 5.7 deg from -z).
#: Compared with `arm/ik.py`'s DEFAULT_Q_REST ([0, -0.3, 0, 0.7, 0, 1.0, 0]):
#: - joint2 is moved from -0.3 to -0.5 rad (further from the joint2=0 shoulder singularity
#:   where joint1 and joint3 align),
#: - joint6 is moved from 1.0 to 1.3 rad (further from the joint6=0 wrist singularity
#:   where joint5 and joint7 align),
#: - every joint sits at least 0.892 rad inside the URDF position limits,
#: - smallest singular value of the 6x7 Jacobian is sigma_min = 0.2089 (vs 0.2092 in
#:   `arm/ik.py`, set by the in-plane radial extension mode at joint4 = 0.7), while the
#:   out-of-plane shoulder/wrist singular value sigma_3 improves from 0.7841 to 0.9497.
DEFAULT_Q_REST = np.array([0.0, -0.5, 0.0, 0.7, 0.0, 1.3, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class StepResult:
    q_next: np.ndarray     # (7,) joint angles in radians, clamped inside [q_min + margin, q_max - margin]
    pos_err: float         # Euclidean position error after the step, metres
    rot_err: float         # Geodesic orientation error ||log3(R_target R^T)|| after the step, radians
    sigma_min: float       # Smallest singular value of the 6x7 Jacobian at the input q
    scaled: bool           # True when dq was uniformly scaled down to respect max_joint_speed * dt


class DiffIK:
    """Single-step damped least-squares differential IK with adaptive damping for xArm7."""

    def __init__(
        self,
        urdf_path: Path | str = DEFAULT_URDF,
        ee_frame: str = "link_eef",
        q_rest: Optional[np.ndarray] = None,
        gain: float = 1.0,
        rot_weight: float = 0.8,
        null_gain: float = 0.05,
        damping: float = 0.10,
        sigma_threshold: float = 0.12,
        max_joint_speed: Optional[float | np.ndarray] = None,
        limit_margin: float = 0.05,
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        if not self.model.existFrame(ee_frame):
            raise ValueError(f"frame {ee_frame!r} not in {self.urdf_path}")
        self.ee_frame = ee_frame
        self.ee_id = self.model.getFrameId(ee_frame)

        self.q_min = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.q_max = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.velocity_limit = np.asarray(self.model.velocityLimit, dtype=np.float64).copy()

        self.limit_margin = float(limit_margin)
        self.q_lo = self.q_min + self.limit_margin
        self.q_hi = self.q_max - self.limit_margin

        self.q_rest = (
            DEFAULT_Q_REST.copy()
            if q_rest is None
            else np.asarray(q_rest, dtype=np.float64).reshape(self.model.nq).copy()
        )
        self.gain = float(gain)
        self.rot_weight = float(rot_weight)
        self.null_gain = float(null_gain)
        self.damping = float(damping)
        self.sigma_threshold = float(sigma_threshold)

        if max_joint_speed is None:
            self.max_joint_speed = self.velocity_limit.copy()
        else:
            self.max_joint_speed = np.broadcast_to(
                np.asarray(max_joint_speed, dtype=np.float64), (self.model.nq,)
            ).copy()

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics of `ee_frame` in the URDF root frame (`link_base`)."""
        q_arr = np.asarray(q, dtype=np.float64).reshape(self.model.nq)
        pin.forwardKinematics(self.model, self.data, q_arr)
        pin.updateFramePlacement(self.model, self.data, self.ee_id)
        oMf = self.data.oMf[self.ee_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def step(
        self,
        q: np.ndarray,
        p_target: np.ndarray,
        R_target: np.ndarray,
        dt: float,
    ) -> StepResult:
        """Take one damped least-squares velocity step toward `(p_target, R_target)`."""
        q_curr = np.clip(
            np.asarray(q, dtype=np.float64).reshape(self.model.nq),
            self.q_lo,
            self.q_hi,
        )
        p_des = np.asarray(p_target, dtype=np.float64).reshape(3)
        R_des = np.asarray(R_target, dtype=np.float64).reshape(3, 3)
        dt_f = float(dt)

        p, R = self.fk(q_curr)
        e_pos = p_des - p
        e_rot = pin.log3(R_des @ R.T)
        twist = self.gain * np.concatenate([e_pos, self.rot_weight * e_rot])

        J = pin.computeFrameJacobian(
            self.model, self.data, q_curr, self.ee_id, pin.LOCAL_WORLD_ALIGNED
        )
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        sigma_min = float(S[-1])

        if self.sigma_threshold > 0.0 and sigma_min < self.sigma_threshold:
            lam2 = (self.damping ** 2) * (1.0 - (sigma_min / self.sigma_threshold) ** 2)
        else:
            lam2 = 0.0

        denom = S ** 2 + lam2
        s_inv = np.where(denom > 0.0, S / denom, 0.0)
        J_pinv = (Vt.T * s_inv) @ U.T

        eye_nv = np.eye(self.model.nv)
        dq = J_pinv @ twist + (eye_nv - J_pinv @ J) @ (self.null_gain * (self.q_rest - q_curr))

        max_dq = self.max_joint_speed * dt_f
        ratios = np.abs(dq) / np.maximum(max_dq, 1e-12)
        peak_ratio = float(ratios.max())
        scaled = bool(peak_ratio > 1.0)
        if scaled:
            dq = dq / peak_ratio

        q_next = np.clip(q_curr + dq, self.q_lo, self.q_hi)
        p_next, R_next = self.fk(q_next)
        pos_err = float(np.linalg.norm(p_des - p_next))
        rot_err = float(np.linalg.norm(pin.log3(R_des @ R_next.T)))

        return StepResult(
            q_next=q_next,
            pos_err=pos_err,
            rot_err=rot_err,
            sigma_min=sigma_min,
            scaled=scaled,
        )
