"""Checks for arm/ik.py: rest posture, joint limits, and FK->IK round-trip convergence."""
import importlib.util

import numpy as np
import pytest

# pinocchio lives in .venv-arm only. A module-level pytest.importorskip would be the usual
# way, but ROS's launch_testing pytest plugin turns that skip into "skip the whole run",
# so the vision suite in the other venv collected nothing. Skip by marker instead.
HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None
pytestmark = pytest.mark.skipif(not HAS_PINOCCHIO, reason="needs pinocchio (.venv-arm)")
if HAS_PINOCCHIO:
    import pinocchio

    from arm.ik import DEFAULT_Q_REST, DEFAULT_URDF, ArmIK


def test_default_q_rest_is_in_working_region_and_far_from_limits():
    ik = ArmIK(DEFAULT_URDF)
    p, R = ik.fk(DEFAULT_Q_REST)

    assert 0.30 <= p[0] <= 0.40
    assert abs(p[1]) < 1e-4
    assert 0.20 <= p[2] <= 0.40
    assert np.isclose(np.linalg.det(R), 1.0)

    margin = np.minimum(DEFAULT_Q_REST - ik.q_min, ik.q_max - DEFAULT_Q_REST)
    assert np.all(margin > 0.8)


def test_fk_ik_at_rest_converges_in_zero_iters():
    ik = ArmIK(DEFAULT_URDF)
    p, R = ik.fk(ik.q_rest)
    res = ik.solve(p, R, ik.q_rest)

    assert res.converged
    assert res.iters == 0
    assert res.pos_err < 1e-12
    assert res.rot_err < 1e-12
    np.testing.assert_allclose(res.q, ik.q_rest)


def test_fk_ik_round_trip_on_random_reachable_poses():
    ik = ArmIK(DEFAULT_URDF)
    rng = np.random.default_rng(0)
    n_samples = 60
    converged_from_perturbed = 0
    converged_from_rest = 0

    for _ in range(n_samples):
        q_true = rng.uniform(ik.q_min, ik.q_max)
        p_target, R_target = ik.fk(q_true)

        q_pert = np.clip(q_true + rng.normal(0.0, 0.25, size=ik.model.nq), ik.q_min, ik.q_max)
        res_pert = ik.solve(p_target, R_target, q_pert)
        assert np.all(res_pert.q >= ik.q_min - 1e-12)
        assert np.all(res_pert.q <= ik.q_max + 1e-12)
        if res_pert.converged:
            converged_from_perturbed += 1
            p_sol, R_sol = ik.fk(res_pert.q)
            assert np.linalg.norm(p_target - p_sol) <= ik.pos_tol
            assert np.linalg.norm(pinocchio.log3(R_target @ R_sol.T)) <= ik.rot_tol

        res_rest = ik.solve(p_target, R_target, ik.q_rest)
        assert np.all(res_rest.q >= ik.q_min - 1e-12)
        assert np.all(res_rest.q <= ik.q_max + 1e-12)
        if res_rest.converged:
            converged_from_rest += 1

    assert converged_from_perturbed / n_samples >= 0.95
    assert converged_from_rest / n_samples >= 0.95
