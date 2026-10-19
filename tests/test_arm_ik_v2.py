"""Checks for arm/ik_v2.py: rest posture, convergence, velocity scaling, limits, and wrist singularity."""
import importlib.util

import numpy as np
import pytest

# pinocchio lives in .venv-arm only. Use marker-based skip (NOT module-level
# pytest.importorskip) so the ROS launch_testing plugin does not abort the vision suite.
HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None
pytestmark = pytest.mark.skipif(not HAS_PINOCCHIO, reason="needs pinocchio (.venv-arm)")
if HAS_PINOCCHIO:
    from arm.ik_v2 import DEFAULT_Q_REST, DEFAULT_URDF, DiffIK


def test_default_q_rest_posture_and_singular_values():
    ik = DiffIK(DEFAULT_URDF)
    p, R = ik.fk(DEFAULT_Q_REST)

    assert 0.30 <= p[0] <= 0.45
    assert abs(p[1]) < 1e-4
    assert 0.20 <= p[2] <= 0.40
    assert np.isclose(np.linalg.det(R), 1.0)
    assert R[2, 2] < -0.95 or R[0, 2] > 0.95

    assert abs(DEFAULT_Q_REST[1]) >= 0.45
    assert abs(DEFAULT_Q_REST[5]) >= 1.0
    margin = np.minimum(DEFAULT_Q_REST - ik.q_min, ik.q_max - DEFAULT_Q_REST)
    assert np.all(margin > 0.8)

    res = ik.step(DEFAULT_Q_REST, p, R, dt=1.0 / 30.0)
    assert not res.scaled
    assert res.pos_err < 1e-12
    assert res.rot_err < 1e-12
    assert res.sigma_min > 0.20
    np.testing.assert_allclose(res.q_next, DEFAULT_Q_REST, atol=1e-12)


def test_repeated_steps_converge_to_static_reachable_targets():
    ik = DiffIK(DEFAULT_URDF)
    rng = np.random.default_rng(7)
    dt = 1.0 / 30.0

    for _ in range(15):
        q_goal = np.clip(
            ik.q_rest + rng.uniform(-0.35, 0.35, size=ik.model.nq),
            ik.q_lo + 0.1,
            ik.q_hi - 0.1,
        )
        p_target, R_target = ik.fk(q_goal)
        q = ik.q_rest.copy()
        for _ in range(80):
            res = ik.step(q, p_target, R_target, dt)
            q = res.q_next
            if res.pos_err < 1e-3 and res.rot_err < 1e-2:
                break
        assert res.pos_err < 1e-3
        assert res.rot_err < 1e-2


def test_per_step_dq_never_exceeds_max_joint_speed_and_preserves_direction():
    dt = 1.0 / 30.0
    for max_speed in (3.14, 1.0):
        ik = DiffIK(DEFAULT_URDF, max_joint_speed=max_speed)
        ik_unbounded = DiffIK(DEFAULT_URDF, max_joint_speed=1e6)
        q_far = np.array([0.5, -0.8, -0.4, 1.2, 0.5, 1.6, -0.5])
        p_target, R_target = ik.fk(q_far)

        q = ik.q_rest.copy()
        saw_scaled = False
        for _ in range(40):
            res_unbounded = ik_unbounded.step(q, p_target, R_target, dt)
            res = ik.step(q, p_target, R_target, dt)
            dq = res.q_next - q
            assert np.all(np.abs(dq) <= max_speed * dt + 1e-12)
            if res.scaled:
                saw_scaled = True
                dq_raw = res_unbounded.q_next - q
                cos_sim = np.dot(dq, dq_raw) / (np.linalg.norm(dq) * np.linalg.norm(dq_raw))
                assert cos_sim > 0.999999
            q = res.q_next
        assert saw_scaled


def test_joint_limits_respected_with_margin():
    margin = 0.05
    dt = 1.0 / 30.0
    ik = DiffIK(DEFAULT_URDF, limit_margin=margin)

    # Start near joint4 and joint6 lower limits and command an unreachable pose further out
    q = ik.q_lo + 0.02
    p_unreachable = np.array([1.2, -0.8, -0.5])
    R_target = np.eye(3)

    for _ in range(60):
        res = ik.step(q, p_unreachable, R_target, dt)
        assert np.all(res.q_next >= ik.q_min + margin - 1e-12)
        assert np.all(res.q_next <= ik.q_max - margin + 1e-12)
        assert np.all(np.abs(res.q_next - q) <= ik.max_joint_speed * dt + 1e-12)
        q = res.q_next


def test_wrist_singularity_traversal_keeps_every_step_bounded():
    dt = 1.0 / 30.0
    max_speed = 1.5
    ik = DiffIK(DEFAULT_URDF, max_joint_speed=max_speed)

    # Start near joint6 = +0.25 and drive target poses across joint6 = 0 to joint6 = -0.25
    # with an out-of-plane rotation component that excites the wrist singularity.
    q_start = np.array([0.0, -0.5, 0.0, 0.9, 0.0, 0.25, 0.0])
    q = q_start.copy()
    min_sigma_seen = float("inf")

    for q6 in np.linspace(0.25, -0.25, 51):
        q_ref = np.array([0.15, -0.5, -0.1, 0.9, 0.2, q6, -0.2])
        p_target, R_target = ik.fk(q_ref)
        res = ik.step(q, p_target, R_target, dt)
        dq = res.q_next - q
        assert np.all(np.isfinite(res.q_next))
        assert np.max(np.abs(dq)) <= max_speed * dt + 1e-12
        assert np.all(res.q_next >= ik.q_lo - 1e-12)
        assert np.all(res.q_next <= ik.q_hi + 1e-12)
        min_sigma_seen = min(min_sigma_seen, res.sigma_min)
        q = res.q_next

    assert min_sigma_seen < ik.sigma_threshold
