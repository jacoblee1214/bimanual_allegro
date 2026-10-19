"""arm/gate.py: impossible wrist motion is dropped, long gaps re-clutch."""
import numpy as np
from scipy.spatial.transform import Rotation

from arm.gate import Verdict, WristGate

DT = 1 / 30
I3 = np.eye(3)


def test_first_sample_engages_and_slow_motion_passes():
    gate = WristGate()
    assert gate(0.0, np.zeros(3), I3) is Verdict.REENGAGE
    for i in range(1, 10):   # 0.3 m/s and 1 rad/s
        R = Rotation.from_rotvec([0, 0, i * DT]).as_matrix()
        assert gate(i * DT, np.array([0.3 * i * DT, 0, 0]), R) is Verdict.ACCEPT


def test_a_jump_is_rejected_and_the_reference_stays():
    gate = WristGate()
    gate(0.0, np.zeros(3), I3)
    assert gate(DT, np.array([0.1, 0, 0]), I3) is Verdict.REJECT          # 3 m/s
    flipped = Rotation.from_rotvec([0.8, 0, 0]).as_matrix()
    assert gate(2 * DT, np.zeros(3), flipped) is Verdict.REJECT           # 12 rad/s over 2 frames
    assert gate(3 * DT, np.array([0.01, 0, 0]), I3) is Verdict.ACCEPT     # back near the last accepted


def test_a_gap_or_a_run_of_rejects_reengages():
    gate = WristGate(reacquire_gap=0.3)
    gate(0.0, np.zeros(3), I3)
    assert gate(0.5, np.array([0.3, 0, 0]), I3) is Verdict.REENGAGE       # hand was gone 0.5 s
    for k in range(1, 9):                                                  # stuck 1 m away
        assert gate(0.5 + k * DT, np.array([1.3, 0, 0]), I3) is Verdict.REJECT
    assert gate(0.5 + 10 * DT, np.array([1.3, 0, 0]), I3) is Verdict.REENGAGE
