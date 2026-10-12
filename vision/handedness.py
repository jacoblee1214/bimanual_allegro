"""
handedness.py — which detected hand is the operator's left and which the right.

docs/01_vision_design.md, 3.2. Kept simple on purpose: labels and one position
rule, no tracking across frames. Check V2 measures how often that is wrong and
decides whether tracking is needed.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

#: MediaPipe decides handedness as if the image were mirrored (a selfie view),
#: and its docs say to swap the labels when it is not. The tracker feeds it the
#: unmirrored image (design doc 3.1), hence the swap. Unconfirmed on this
#: camera until check V1.
MEDIAPIPE_LABEL_TO_SIDE = {"Left": "right", "Right": "left"}

MIN_SCORE = 0.8


class Detection(NamedTuple):
    label: str      # MediaPipe's handedness label, "Left" or "Right"
    score: float    # its handedness score
    wrist_x: float  # landmark 0's x over image width, in the unmirrored image


def assign_sides(detections: Sequence[Detection], min_score: float = MIN_SCORE) -> dict[str, int]:
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
        side_a = MEDIAPIPE_LABEL_TO_SIDE[detections[a].label]
        side_b = MEDIAPIPE_LABEL_TO_SIDE[detections[b].label]
        if side_a != side_b:
            return {side_a: a, side_b: b}
        right, left = sorted(kept, key=lambda i: detections[i].wrist_x)
        return {"right": right, "left": left}

    if len(kept) == 1:
        i = kept[0]
        return {MEDIAPIPE_LABEL_TO_SIDE[detections[i].label]: i}

    return {}
