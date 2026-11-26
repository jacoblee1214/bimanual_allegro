"""vision/rotation_hold.py: the last sensor rotation is repeated, for a limited number of frames."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vision.rotation_hold import hold_rotation


@dataclass(frozen=True)
class Fused:
    source: str
    rotation: Optional[np.ndarray] = None


class Scripted:
    def __init__(self):
        self.calls, self.resets = 0, 0

    def reset(self):
        self.resets += 1

    def __call__(self, rotation):
        self.calls += 1
        return Fused("depth" if rotation is not None else "hold", rotation)


def test_holds_then_gives_up_and_resets():
    fusion = hold_rotation(Scripted, max_frames=2)()
    R = np.eye(3) * 2.0
    assert fusion(None).rotation is None                 # nothing to hold yet
    assert fusion(R).rotation is R
    held = fusion(None)
    assert held.rotation is R and held.source == "hold"  # the source stays honest
    assert fusion(None).rotation is R
    assert fusion(None).rotation is None                 # max_frames used up
    fusion(R)
    fusion.reset()
    assert fusion(None).rotation is None and fusion.resets == 1 and fusion.calls == 7


def test_limit_rotation():
    import pytest
    pytest.importorskip("pinocchio")
    from scipy.spatial.transform import Rotation
    from arm.arm_follow_node_v7 import limit_rotation
    start = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    small = Rotation.from_euler("z", 20, degrees=True).as_matrix() @ start
    big = Rotation.from_euler("z", 100, degrees=True).as_matrix() @ start
    assert np.allclose(limit_rotation(small, start, np.radians(45)), small)
    limited = limit_rotation(big, start, np.radians(45))
    assert np.allclose(limited, Rotation.from_euler("z", 45, degrees=True).as_matrix() @ start)
