"""vision/filters.py: the two properties the One Euro filter is chosen for."""
import numpy as np

from vision.filters import OneEuroFilter

RATE = 30.0


def run(f, xs):
    return np.array([f(np.atleast_1d(x), i / RATE)[0] for i, x in enumerate(xs)])


def test_first_sample_passes_through_and_a_constant_stays_put():
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    out = run(f, np.full(30, 0.42))
    np.testing.assert_allclose(out, 0.42)


def test_a_still_signal_loses_most_of_its_jitter():
    noise = np.random.default_rng(0).normal(0.0, 0.003, 300)  # 3 mm at 30 Hz
    out = run(OneEuroFilter(min_cutoff=1.0, beta=10.0), noise)
    assert out[30:].std() < 0.4 * noise[30:].std()


def test_speed_raises_the_cutoff_so_a_moving_signal_lags_less():
    ramp = np.arange(90) / RATE * 0.5  # 0.5 m/s
    lag_fixed = ramp[-1] - run(OneEuroFilter(min_cutoff=1.0, beta=0.0), ramp)[-1]
    lag_adaptive = ramp[-1] - run(OneEuroFilter(min_cutoff=1.0, beta=10.0), ramp)[-1]
    assert lag_adaptive < 0.5 * lag_fixed


def test_reset_forgets_the_old_position():
    f = OneEuroFilter(min_cutoff=1.0)
    run(f, np.zeros(10))
    f.reset()
    assert f(np.array([5.0]), 100.0)[0] == 5.0
