"""The left/right rules of vision/handedness.py (design doc 3.2)."""
from vision.handedness import Detection, assign_sides

# Unmirrored image: the operator's right hand shows on the left (small x), and
# MediaPipe labels it "Left".
RIGHT = Detection("Left", 0.95, 0.3)
LEFT = Detection("Right", 0.95, 0.7)


def test_labels_are_swapped_for_the_unmirrored_image():
    assert assign_sides([RIGHT, LEFT]) == {"right": 0, "left": 1}
    assert assign_sides([LEFT, RIGHT]) == {"left": 0, "right": 1}


def test_one_hand_only_publishes_its_own_side():
    assert assign_sides([RIGHT]) == {"right": 0}
    assert assign_sides([LEFT]) == {"left": 0}


def test_unsure_hands_are_dropped():
    assert assign_sides([RIGHT._replace(score=0.6)]) == {}
    assert assign_sides([RIGHT._replace(score=0.6), LEFT]) == {"left": 1}
    assert assign_sides([RIGHT._replace(score=0.6), LEFT._replace(score=0.5)]) == {}


def test_same_label_twice_is_settled_by_position():
    both_left_labelled = [Detection("Left", 0.9, 0.8), Detection("Left", 0.9, 0.2)]
    assert assign_sides(both_left_labelled) == {"left": 0, "right": 1}


def test_labels_win_over_position_when_hands_cross():
    crossed = [RIGHT._replace(wrist_x=0.7), LEFT._replace(wrist_x=0.3)]
    assert assign_sides(crossed) == {"right": 0, "left": 1}


def test_nothing_detected():
    assert assign_sides([]) == {}
