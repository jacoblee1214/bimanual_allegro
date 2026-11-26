"""show/rps.py: gestures from landmarks, and a round played against a scripted hand."""
import numpy as np
import pytest

from show import rps

SAFE = {  # the hand controller's limits that the poses must respect (safety_utils.py, both hands)
    "thumb": [(-0.035, 1.658), (-1.658, 1.658), (-0.175, 1.309), (-0.175, 1.396)],
    "finger": [(-0.384, 0.384), (-0.070, 1.571), (-0.175, 1.396), (-0.175, 1.396)],
}


def hand(straight):
    """21 landmarks of a hand whose index, middle, ring, pinky are straight or curled."""
    p = np.zeros((21, 3))
    for f, (x, is_straight) in enumerate(zip((0.03, 0.01, -0.01, -0.03), straight)):
        knuckle = np.array([x, 0.09, 0.0])
        tip = knuckle + ([0.0, 0.075, 0.0] if is_straight else [0.0, -0.01, -0.03])
        p[5 + 4 * f], p[8 + 4 * f] = knuckle, tip
    return p.ravel()


FIST, OPEN, VEE = hand([0, 0, 0, 0]), hand([1, 1, 1, 1]), hand([1, 1, 0, 0])


def test_classify():
    assert rps.classify(rps.extension(FIST)) == rps.ROCK
    assert rps.classify(rps.extension(OPEN)) == rps.PAPER
    assert rps.classify(rps.extension(VEE)) == rps.SCISSORS
    assert rps.classify(rps.extension(hand([1, 0, 0, 0]))) is None
    assert rps.classify(rps.extension(3.0 * OPEN)) == rps.PAPER      # hand size does not matter


def test_poses_within_limits():
    for name, q in rps.POSES.items():
        limits = SAFE["thumb"] + SAFE["finger"] * 4
        assert all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)), name


def play(mode, throw, seed=0):
    game = rps.Game(np.random.default_rng(seed), mode=mode)
    game.start(0.0)
    shown = []
    for i in range(int(8.0 * 30)):
        t = i / 30.0
        opening = np.clip((t - (game.shoot_at - 0.1)) / 0.2, 0.0, 1.0)     # the fist opens over 0.2 s
        ext = rps.extension((1 - opening) * FIST + opening * throw)
        game.step(t, ext)
        shown.append((t, game.phase, game.robot))
    return game, shown


@pytest.mark.parametrize("throw,name", [(FIST, rps.ROCK), (OPEN, rps.PAPER), (VEE, rps.SCISSORS)])
def test_fair_round(throw, name):
    game, shown = play("fair", throw)
    assert game.human == name and game.phase == "mirror"
    assert sum(game.score.values()) == 1
    assert all(robot == rps.ROCK for t, phase, robot in shown if phase in ("ready",))
    first = next(t for t, _, robot in shown if robot not in (rps.ROCK, None))
    assert abs(first - (game.shoot_at - game.lead)) < 0.05


@pytest.mark.parametrize("throw", [FIST, OPEN, VEE])
def test_reader_wins(throw):
    game, _ = play("reader", throw)
    assert game.winner == "robot" and game.score["robot"] == 1
    if throw is not FIST:
        assert game.react_ms is not None and game.react_ms < 200.0


def test_no_hand_is_no_throw():
    game = rps.Game(np.random.default_rng(0))
    game.start(0.0)
    for i in range(300):
        game.step(i / 30.0, None)
    assert game.winner == "no throw" and sum(game.score.values()) == 0


def test_pump_is_three_pumps_and_bounded():
    game = rps.Game(np.random.default_rng(0))
    game.start(0.0)
    values = []
    for i in range(600):
        game.step(i / 100.0, None)
        values.append(game.pump(i / 100.0))
    values = np.array(values)
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert ((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] > 0.9)).sum() == 3
    assert np.abs(np.diff(values)).max() * 100.0 <= np.pi / game.period + 1e-6    # peak rate, 1/s
