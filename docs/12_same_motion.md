# 12. The robot moves as the operator moves

2026-11-26. The teleop of docs/10 ran with every part real (data/runs/t5), but the robot does
not move as the operator does. The differences, measured in t5, and the order to close them:

| difference | now | evidence |
|---|---|---|
| arm travel | half the hand's motion, within 7 cm | both arms reached the 70 mm cap |
| wrist rotation | not followed | `--position-only` |
| arm speed | 0.3 rad/s per joint | follow commands of 0.37 to 0.42 rad/s were cut |
| fingers | measured minus commanded, p95 33 deg | tools/analyze_run.py |
| right-hand tracking | depth 65 %, held 35 %; jitter 10 to 16 mm RMS | hand 0.18 m from the camera |

Order: (1) camera at head height, (2) arm travel 1:1, (3) wrist rotation, (4) arm speed and
lag, (5) finger error.

## Step 1: camera

At 14:5x the camera read pitch 30.7 deg, USB 3.2. It looks at the operator's back from about
1.6 m: the hands are behind the body and too far. It has to come forward to just above and
behind the head, so that the hands are 0.4 to 0.6 m from it. Not done yet.

## Step 2: arm travel 1:1, simulated

`tools/eval_reach_from_home.py` (the follow node's IK, position only, from the home pose, no
joint box; data/results/reach_from_home.json):

| radius | directions missed (of 26) | worst error | smallest singular value | largest joint offset | EEF lowest z |
|---|---|---|---|---|---|
| 7 cm | 0 | 0.0 mm | 0.136 / 0.129 | 0.32 rad | 0.266 m |
| 15 cm | 0 | 2.1 / 2.3 mm | 0.080 / 0.077 | 0.82 rad | 0.186 m |
| 25 cm | 6 / 5 | 57 / 55 mm (forward) | 0.015 | 1.44 rad | 0.086 m |

(left / right arm.) So from the home pose 15 cm works and 25 cm does not: forward, the arm is
nearly straight. More than 15 cm needs a start pose with the elbow more bent.

At 15 cm the singular value falls below the IK's damping threshold (0.12), so in some
directions the IK slows itself down; the position error stays at about 2 mm.

Settings: `config/teleop_same_15cm.yaml` (scale 1.0, radius 0.15 m, joint box 0.9 rad),
used with `tools/run_teleop_real_v2.sh --config config/teleop_same_15cm.yaml`.

Replay of the clip `ego_depth_work` through the whole pipeline with these settings (arms
simulated from the home pose, hands simulated, ROS domain 77):

| arm speed cap | EEF travel, left / right | largest joint offset | joint speed p95 |
|---|---|---|---|
| 0.3 rad/s | 148 / 149 mm | 0.82 rad | 0.29 / 0.30 rad/s |
| 1.0 rad/s | 148 / 150 mm | 0.82 rad | 0.48 / 0.64 rad/s |

## Step 4, looked at early: speed

The same tool swings the target +-radius at 0.5 Hz (peak hand speed 0.47 m/s at 15 cm):

| radius | cap | worst tracking error (x, y, z) | lag (x, y, z) |
|---|---|---|---|
| 7 cm | 0.3 rad/s | 65, 58, 46 mm | 320, 290, 200 ms |
| 7 cm | 1.0 rad/s | 0, 0, 0 mm | 0 ms |
| 15 cm | 0.3 rad/s | 159, 155, 149 mm | 400, 380, 360 ms |
| 15 cm | 0.6 rad/s | 146, 128, 108 mm | 330, 300, 220 ms |
| 15 cm | 1.0 rad/s | 121, 70, 26 mm | 220, 120, 30 ms |

At 1:1 the 0.3 rad/s cap is what the eye will see: the arm arrives 0.4 s late on a brisk
swing. Slow motions (t5's hand speed: median 0.03 to 0.04 m/s, p95 0.15 to 0.2 m/s) are
followed. These numbers are the IK and the cap only; camera and robot latency come on top.

## Not known, and needed before the real arms run at 15 cm

1. The distance between the two arm bases. config/arms_sim.yaml says 1.5 m, "not measured".
   At 15 cm each EEF moves up to 15 cm towards the other arm, and the hands are wider than
   the flange. With bases closer than about 0.8 m the two hands can meet.
2. The table height relative to the arm bases. At 15 cm the flange comes down to 0.186 m
   above the base plane (0.336 m at the start), and the hand hangs below the flange.
3. Jitter at 1:1. Left wrist in t5: 1 to 4 mm RMS. The right was 10 to 16 mm with the hand
   too close; it needs a run with the camera in its new place.

## Order on the real robot (each step asks before moving)

    C=config/teleop_same_15cm.yaml
    tools/run_teleop_real_v2.sh s1 --config $C                                  # arms read only: check tracking, jitter
    tools/run_teleop_real_v2.sh s2 --config $C --arms real                      # 1:1, 15 cm, 0.3 rad/s
    tools/run_teleop_real_v2.sh s3 --config $C --arms real --arm-speed 0.6
    tools/run_teleop_real_v2.sh s4 --config $C --arms real --arm-speed 1.0 --hands both
