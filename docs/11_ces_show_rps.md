# 11. CES show: rock-paper-scissors against the robot

2026-11-26. A feature for visitors at CES, on top of the teleop of docs/10.

## Why this one

A booth visitor gives a demo about ten seconds. Rock-paper-scissors needs no explanation in
any language, takes ten seconds, has a winner, and is played with exactly what is on show:
a hand that can make shapes. The visitor needs nothing on their hands; one camera sees them.
Until someone plays, the robot keeps mirroring the visitor, which is the existing teleop.

A fast robot hand that always wins this game has been shown before, with a 1 ms camera
(Katsuki, Yamakawa, Ishikawa, "High-speed Human / Robot Hand Interaction System", HRI 2015
Extended Abstracts). This setup has a 30 fps camera and a 3 rad/s cap on the fingers, so
the `reader` mode below is a slower relative of it and its lag will be visible.

## How a round goes

| time | screen | robot |
|---|---|---|
| a V sign held 1 s (or key G, or `/show/start`) | GET READY | arms hold; right hand makes a fist; left hand rests |
| +1.5 s, three counts of 1 s | ROCK, PAPER, SCISSORS | right elbow pumps once per count (0.08 rad, peak 0.25 rad/s) |
| +4.5 s | SHOOT! | right hand throws |
| visitor's hand steady for 4 frames | ROBOT WINS / YOU WIN! / DRAW, score | left hand opens and closes if the robot won |
| +3 s | back to mirroring | the bridges bring arms and hands back at their speed caps |

Modes: `fair` (the throw is drawn before the count and starts 0.25 s before SHOOT) and
`reader` (the robot watches the visitor's fist open, decides after two frames that agree,
and throws what beats it; the screen shows the time from the first finger motion to the
decision). Key M switches between them.

## How it is built

- `show/rps.py`: gestures from the tracker's 21 world landmarks and the round as a state
  machine, without ROS. A finger is extended when tip-to-wrist over knuckle-to-wrist is
  above 1.35. In the recorded clips that ratio is 1.5 to 2.0 for straight fingers and 0.8
  to 1.1 for a fist (data/results/m5_detections_*.npz).
- `show/rps_node.py`: sits between the teleop and the bridges. It passes the retargeted hand
  joints and the arm follow's joints on unchanged while nobody plays and replaces them
  during a round. The speed caps and the joint box stay in the bridges.
- `launch/teleop_real_v2.launch.py show:=rps`, `tools/run_teleop_real_v2.sh <tag> --show rps`.
- Robot hand poses: the fist is what the real left hand reached in data/runs/t5 (thumb
  0.63, -1.38, 0.74, 0.96; fingers 1.3, 1.3, 1.3 rad), paper is all zeros, scissors mixes
  the two per finger.

## What was checked (no robot moved)

- `tests/test_rps.py`, 10 tests: gestures, poses within the hand's limits, a fair round, the
  reader winning against a scripted hand, no hand, the pump's shape.
- A round in each mode with simulated arms and hands and the clip `ego_depth_work` (ROS
  domain 77, started automatically): every phase in order, the result reported, back to
  mirroring. Right hand command reached 1.3 rad; hand command speed p99 3.0 rad/s; elbow
  stroke -0.080 rad.

## What is not checked yet

1. The three poses on the real hands. The fist was reached by the left hand only, with the
   thumb following the retargeting; the right hand has not made it. Scissors was never made.
2. Gestures of a real visitor in front of the camera: the clips have open hands and fists
   but no rock-paper-scissors throws. The `reader` thresholds (1.3 moving, 1.2 still) are
   set from those clips, not from throws.
3. `reader` timing. Estimate: two frames (67 ms) plus camera latency, then up to 0.43 s for
   a finger to travel 1.3 rad at 3 rad/s.
4. When a round ends, the arm follow's target can be up to a few hundredths of a radian from
   where the arm held (0.053 rad in the simulated round). The arm bridge closes that at
   0.3 rad/s; with `arms:=sim` there is no bridge and RViz shows a small jump.
5. Other people's hands behind the visitor. The tracker takes the two hands it finds.

## Order to try it

    tools/run_teleop_real_v2.sh g1 --show rps                              # nothing moves: gestures, screen
    tools/run_teleop_real_v2.sh g2 --show rps --hands right                # the real right hand throws
    tools/run_teleop_real_v2.sh g3 --show rps --arms real --hands both     # everything
    ... --rps-mode reader

## Next candidates

An idle routine and a clean hand-over between visitors; "teach it in ten seconds" (record a
visitor's motion, replay it smoothed: stages 4 and 5 in small); picking up a prize, once the
arm's 7 cm radius is widened.
