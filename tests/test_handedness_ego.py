"""vision/handedness_ego.py: position decides in the egocentric view."""
from vision.handedness import Detection
from vision.handedness_ego import EgoSides


def det(x: float, label: str = "Right") -> Detection:
    return Detection(label=label, score=0.9, wrist_x=x)


def test_two_hands_go_by_position_whatever_the_labels_say():
    sides = EgoSides()
    assert sides([det(0.7, "Left"), det(0.2, "Left")], 0.0) == {"left": 1, "right": 0}
    assert sides([det(0.1, "Right"), det(0.8, "Right")], 0.1) == {"left": 0, "right": 1}


def test_one_hand_alone_goes_by_the_half_it_is_in():
    sides = EgoSides()
    assert sides([det(0.7)], 0.0) == {"right": 0}
    sides.reset()
    assert sides([det(0.3, "Right")], 0.0) == {"left": 0}


def test_a_hand_near_the_middle_keeps_its_side():
    sides = EgoSides(memory_s=0.5)
    sides([det(0.45), det(0.9)], 0.0)
    assert sides([det(0.52)], 0.1) == {"left": 0}       # past the midline, still the left hand
    assert sides([det(0.55)], 1.0) == {"right": 0}      # memory gone: by the half again


def test_nothing_found_assigns_nothing():
    assert EgoSides()([], 0.0) == {}
