"""
handedness.py — which detected hand is the operator's left and which the right.

docs/01_vision_design.md, 3.2. Labels and one position rule, no tracking
across frames. Check V2 (2026-10-12) found no need for more: with both hands
inside the middle 60% of the image, no label came out wrong, palms or backs to
the camera, and the labels kept the sides right through crossings, where
position alone would not. Near the image edges a third of the labels were
wrong, at high scores, so that is a camera placement problem, not one this
rule can fix.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

#: MediaPipe decides handedness as if the image were mirrored (a selfie view),
#: and its docs say to swap the labels when it is not. The tracker feeds it the
#: unmirrored image (design doc 3.1), hence the swap. Confirmed on the HP
#: webcam, 2026-10-12 (design doc check V1): no frame came out the other way.
MEDIAPIPE_LABEL_TO_SIDE = {"Left": "right", "Right": "left"}

#: MediaPipe's newer Tasks API (HandLandmarker) labels the other way round:
#: on the same unmirrored clips its labels are the operator's actual sides,
#: every one of 530 two-hand detections (design doc, V5 GPU check). Use the
#: map that goes with the API the labels came from.
TASKS_LABEL_TO_SIDE = {"Left": "left", "Right": "right"}

MIN_SCORE = 0.8


class Detection(NamedTuple):
    label: str      # MediaPipe's handedness label, "Left" or "Right"
    score: float    # its handedness score
    wrist_x: float  # landmark 0's x over image width, in the unmirrored image


def assign_sides(
    detections: Sequence[Detection],
    min_score: float = MIN_SCORE,
    label_to_side: dict[str, str] = MEDIAPIPE_LABEL_TO_SIDE,
) -> dict[str, int]:
    """{"left": i, "right": j} into `detections`, for the hands that can be told apart.

    A side with no confident hand is left out, so its topic goes quiet rather
    than repeating an old pose.

    * Hands under `min_score` are dropped.
    * Two hands with different labels: the labels win, even when the hands
      cross over and their positions say otherwise.
    * Two hands with the same label: position decides. An operator facing the
      camera has their right hand on the left of the unmirrored image.
    """
    kept = [i for i, d in enumerate(detections) if d.score >= min_score]

    if len(kept) == 2:
        a, b = kept
        side_a = label_to_side[detections[a].label]
        side_b = label_to_side[detections[b].label]
        if side_a != side_b:
            return {side_a: a, side_b: b}
        right, left = sorted(kept, key=lambda i: detections[i].wrist_x)
        return {"right": right, "left": left}

    if len(kept) == 1:
        i = kept[0]
        return {label_to_side[detections[i].label]: i}

    return {}
