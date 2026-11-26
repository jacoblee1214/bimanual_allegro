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

## Real runs at 15 cm (2026-11-26)

| run | settings | result |
|---|---|---|
| s5 | 1:1, 15 cm, 0.3 rad/s, both hands | no error; EEF at the cap 19 to 23 % of the time; the hands moved 58 to 62 cm vertically; hand -> measured EEF lag 0.16 to 0.36 s (rough, correlation 0.3 to 0.6); right bridge speed-limited on 40 % of ticks; left hand tracked 75 % of the time, right 91 % |
| s6 | the same at 0.6 rad/s | no error; bridge speed-limited on 6.5 % of ticks; tracking poor: left 287 wrist samples in 60 s, right hand 0.23 m from the camera with 32 % held |

## The hands are as far apart as the bases: start turned inwards

Each arm follows its own hand's displacement from its own start. From the home pose the
flanges are as far apart as the bases, 0.98 m (between the base cables, user), while the
operator's hands are 0.3 to 0.4 m apart, and the camera sees about 0.67 m across at 0.4 m.
The robot's hands can then never meet, and the operator cannot spread their hands that far
inside the picture.

`tools/find_center_start.py` looks for a start with the flanges 0.3 or 0.4 m apart, at the
home pose's flange orientation (data/results/center_start.json). With x 0.35 m every
direction is reachable at 25 cm (worst 4 to 6 mm); with x 0.45 m two to four directions are
missed at 25 cm. Chosen: flanges 0.40 m apart, x 0.35, z 0.35 m:

| | left | right |
|---|---|---|
| joints, deg | -20.3, -4.0, -21.9, 48.3, 120.7, 50.3, 41.7 | 4.0, -4.7, 37.8, 48.4, 239.4, 49.1, 139.6 |
| smallest singular value at the pose | 0.198 | 0.166 |
| 20 cm: missed / worst error | 0 of 26 / 1.3 mm | 0 of 26 / 1.6 mm |
| 20 cm: largest joint offset / margin to limits | 0.82 / 0.42 rad | 0.85 / 0.44 rad |
| 20 cm: EEF lowest z | 0.15 m | 0.15 m |

The joint-space move from the home pose keeps the flange between z 0.336 and 0.356 m and each
arm on its own side (world y 0.20 to 0.49 m and -0.49 to -0.20 m).

`config/teleop_center.yaml`: that start, 1:1, radius 20 cm, joint box 0.9 rad, 0.3 rad/s.
`arm/arm_follow_node_v7.py --y-range` keeps each flange `hand_gap / 2` (0.11 m) from the middle,
so the flanges stay at least 0.22 m apart. The 0.22 m is a guess, not measured on the hands.

Replay of `ego_depth_work` from that start (simulated, domain 77): EEF travel 192 / 200 mm,
flange distance 0.400 m at the start, 0.307 m at the least, joints within 0.61 rad.

Not checked: the pose and the move on the real arms; whether the forearms or the hands'
cables clear each other and the table at x 0.35 m; whether 0.22 m is enough for two hands.

    C=config/teleop_center.yaml
    tools/run_teleop_real_v2.sh c1 --config $C                    # offers the move to the start; arms read only
    tools/run_teleop_real_v2.sh c2 --config $C --arms real
    tools/run_teleop_real_v2.sh c3 --config $C --arms real --hands both --arm-speed 0.6
