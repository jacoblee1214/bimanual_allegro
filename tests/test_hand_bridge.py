import numpy as np

from hand.hand_bridge_node import limit_speed


def test_limit_speed_caps_each_joint_and_reaches_the_target():
    sent, wanted = np.zeros(3), np.array([0.01, -1.0, 0.5])
    step = limit_speed(sent, wanted, 0.03)
    assert np.allclose(step, [0.01, -0.03, 0.03])
    for _ in range(40):
        sent = limit_speed(sent, wanted, 0.03)
    assert np.allclose(sent, wanted)
