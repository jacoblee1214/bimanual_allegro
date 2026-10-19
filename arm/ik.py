"""
ik.py — damped least-squares 6D inverse kinematics for the 7-DOF xArm7.

The 6D task (position + orientation) leaves one redundant degree of freedom on
a 7-DOF arm. Without a secondary objective the elbow drifts in the null space
over a trajectory; projecting a pull towards `q_rest` into `(I - J^+ J)` keeps
the posture centred and away from joint limits while leaving the EEF pose task
untouched. See docs/03_arm_follow_design.md, section 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "data" / "models" / "xarm7.urdf"

#: Working posture with the EEF at (0.3696, 0.0, 0.3456) m in link_base and
#: tool axes diag(1, -1, -1) (pointing straight down, x forward). Every joint
#: sits at least 0.89 rad (51 deg, joint4 lower limit) inside the URDF limits
#: and away from the joint4=0 and joint6=0 singularities.
DEFAULT_Q_REST = np.array([0.0, -0.3, 0.0, 0.7, 0.0, 1.0, 0.0], dtype=np.float64)

#: Null-space gain pulling toward q_rest per iteration, kept small so the damped
#: projector does not fight the 1 mm position tolerance near workspace edges.
NULL_GAIN = 1e-2

#: Maximum joint-space step norm (rad) in a single Newton iteration, so cold
#: starts far from the target do not slam directly into joint limits.
MAX_ITER_STEP = 1.0


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray          # (7,) joint angles in radians, clamped to URDF limits
    converged: bool        # True when pos_err <= pos_tol and rot_err <= rot_tol
    pos_err: float         # Euclidean position error, metres
    rot_err: float         # Geodesic orientation error ||log3(R_target R^T)||, radians
    iters: int             # Iterations taken


class ArmIK:
    """Pinocchio damped least-squares IK for xArm7."""

    def __init__(
        self,
        urdf_path: Path | str = DEFAULT_URDF,
        ee_frame: str = "link_eef",
        q_rest: Optional[np.ndarray] = None,
        damping: float = 1e-2,
        max_iters: int = 100,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
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

        self.q_rest = (
            DEFAULT_Q_REST.copy()
            if q_rest is None
            else np.asarray(q_rest, dtype=np.float64).reshape(self.model.nq).copy()
        )
        self.damping = float(damping)
        self.max_iters = int(max_iters)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics of `ee_frame` in the URDF root frame (`link_base`)."""
        q_arr = np.asarray(q, dtype=np.float64).reshape(self.model.nq)
        pin.forwardKinematics(self.model, self.data, q_arr)
        pin.updateFramePlacement(self.model, self.data, self.ee_id)
        oMf = self.data.oMf[self.ee_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def solve(
        self,
        p_target: np.ndarray,
        R_target: np.ndarray,
        q_init: np.ndarray,
    ) -> IKResult:
        """Solve for joint angles reaching `(p_target, R_target)` from `q_init`."""
        p_des = np.asarray(p_target, dtype=np.float64).reshape(3)
        R_des = np.asarray(R_target, dtype=np.float64).reshape(3, 3)
        q = np.clip(np.asarray(q_init, dtype=np.float64).reshape(self.model.nq), self.q_min, self.q_max)

        reg = (self.damping ** 2) * np.eye(6)
        eye_nv = np.eye(self.model.nv)

        for it in range(self.max_iters):
            p, R = self.fk(q)
            e_pos = p_des - p
            e_rot = pin.log3(R_des @ R.T)
            pos_err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot))
            if pos_err <= self.pos_tol and rot_err <= self.rot_tol:
                return IKResult(q=q, converged=True, pos_err=pos_err, rot_err=rot_err, iters=it)

            err = np.concatenate([e_pos, e_rot])
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
            J_pinv = J.T @ np.linalg.solve(J @ J.T + reg, np.eye(6))
            dq = J_pinv @ err + (eye_nv - J_pinv @ J) @ (NULL_GAIN * (self.q_rest - q))

            q_next = np.clip(q + dq, self.q_min, self.q_max)
            clamped = (q_next == self.q_min) | (q_next == self.q_max)
            if np.any(clamped):
                J_free = J.copy()
                J_free[:, clamped] = 0.0
                J_free_pinv = J_free.T @ np.linalg.solve(J_free @ J_free.T + reg, np.eye(6))
                dq_clamped = q_next - q
                dq = dq_clamped + J_free_pinv @ (err - J @ dq_clamped)

            step_norm = float(np.linalg.norm(dq))
            if step_norm > MAX_ITER_STEP:
                dq *= MAX_ITER_STEP / step_norm
            q = np.clip(q + dq, self.q_min, self.q_max)

        p, R = self.fk(q)
        pos_err = float(np.linalg.norm(p_des - p))
        rot_err = float(np.linalg.norm(pin.log3(R_des @ R.T)))
        converged = pos_err <= self.pos_tol and rot_err <= self.rot_tol
        return IKResult(q=q, converged=converged, pos_err=pos_err, rot_err=rot_err, iters=self.max_iters)
