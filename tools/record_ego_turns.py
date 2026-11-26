#!/usr/bin/env python3
"""
record_ego_turns.py — tools/record_ego_depth.py with phases for the wrist's rotation and for grasping.

For docs/13_wrist_rotation.md: in the run r1 the palm frame jumped by about 55 degrees about
the finger direction while the hands were being flipped, and the hands were found in only 35
to 43 % of the frames. A clip (colour + depth) lets every way of estimating the orientation
be compared on the same frames, offline. Camera only: no ROS, no robot.

  4 s  get ready: both hands in view, palms DOWN, fingers forward
  4 s  hold still
 12 s  flip: palms DOWN -> UP -> DOWN, slowly, 3 times
 10 s  bend: wrists UP and DOWN, slowly, 3 times (palms down)
 10 s  turn: hands LEFT and RIGHT, slowly, 3 times (palms down)
 10 s  grasp: open, fist, open, pinch (thumb + index), open, repeated

Same files as record_ego_depth.py, in data/clips/<name>/ (default name: ego_turns).

Usage (with .venv-vision):
    tools/record_ego_turns.py [--name ego_turns]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_ego_depth as base  # noqa: E402

PHASES = (("ready", 4.0, "get ready: BOTH hands, palms DOWN, fingers forward"),
          ("still", 4.0, "hold STILL"),
          ("flip", 12.0, "FLIP: palms DOWN - UP - DOWN, slowly, 3 times"),
          ("bend", 10.0, "BEND: wrists UP and DOWN, slowly, 3 times"),
          ("turn", 10.0, "TURN: hands LEFT and RIGHT, slowly, 3 times"),
          ("grasp", 10.0, "GRASP: open, fist, open, pinch, open ..."))

if __name__ == "__main__":
    base.PHASES = PHASES
    base.__doc__ = __doc__
    if "--name" not in sys.argv:
        sys.argv += ["--name", "ego_turns"]
    base.main()
