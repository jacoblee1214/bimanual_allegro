#!/usr/bin/env python3
"""
check_pose.py — is each arm in a pose the wrist-following IK can work from? Reads only.

The R2 dry run (docs/results/r1_left_joint_nudge.md, 3) found both arms resting near the
shoulder singularity: joint 2 near 0 with joints 1 and 3 wound opposite ways, smallest
Jacobian singular value 0.035. There, a few cm of hand motion asks for half a radian
of joint motion. This reads the encoders and reports, per arm:

  sigma_min        smallest singular value of the 6x7 Jacobian (URDF) -- want > 0.15
  |joint2|, |joint6|  distance from the shoulder and wrist singularities -- want > 25, 30 deg
  limit margin     closest joint to its limit -- want > 30 deg
  TCP              where the controller says the tool is (mm, base frame)

Usage (with .venv-arm):
    tools/check_pose.py                 # both arms from config/robots.local.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from xarm.wrapper import XArmAPI

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from arm.ik_v2 import DiffIK  # noqa: E402

WANT = {"sigma_min": 0.15, "joint2_deg": 25.0, "joint6_deg": 30.0, "margin_deg": 30.0}


def check(ik: DiffIK, q: np.ndarray) -> dict:
    J = pin.computeFrameJacobian(ik.model, ik.data, q, ik.ee_id, pin.LOCAL_WORLD_ALIGNED)
    margin = np.minimum(q - ik.q_min, ik.q_max - q)
    return {
        "sigma_min": float(np.linalg.svd(J, compute_uv=False)[-1]),
        "joint2_deg": float(np.degrees(abs(q[1]))),
        "joint6_deg": float(np.degrees(abs(q[5]))),
        "margin_deg": float(np.degrees(margin.min())),
        "margin_joint": int(margin.argmin()) + 1,
    }


def main() -> None:
    ik = DiffIK()
    arms = yaml.safe_load((REPO / "config" / "robots.local.yaml").read_text())["xarm"]
    for side, ip in arms.items():
        arm = XArmAPI(ip, is_radian=True)
        try:
            code, q = arm.get_servo_angle(is_radian=True, is_real=True)
            if code != 0:
                print(f"{side}: could not read joints (code {code})")
                continue
            q = np.array(q[:7])
            r = check(ik, q)
            ok = {k: r[k] > v for k, v in WANT.items()}
            print(f"{side}: state {arm.state}, error {arm.error_code}")
            print(f"  joints deg  {np.round(np.degrees(q), 1).tolist()}")
            print(f"  TCP mm      {np.round(arm.position[:3], 1).tolist()}")
            print(f"  sigma_min   {r['sigma_min']:.3f}  {'ok' if ok['sigma_min'] else 'LOW (want > 0.15)'}")
            print(f"  |joint2|    {r['joint2_deg']:.1f} deg  {'ok' if ok['joint2_deg'] else 'near the shoulder singularity'}")
            print(f"  |joint6|    {r['joint6_deg']:.1f} deg  {'ok' if ok['joint6_deg'] else 'near the wrist singularity'}")
            print(f"  margin      {r['margin_deg']:.1f} deg (joint{r['margin_joint']})  "
                  f"{'ok' if ok['margin_deg'] else 'close to a limit'}")
            print(f"  -> {'READY' if all(ok.values()) else 'not ready'}")
        finally:
            arm.disconnect()


if __name__ == "__main__":
    main()
