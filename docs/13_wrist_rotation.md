# 13. Following the wrist's rotation

2026-11-26. Step 3 of docs/12_same_motion.md. Simulated only; nothing here has run on the arms.

## Where the orientation comes from, and how steady it is

/<side>/wrist_pose carries the palm frame. On frames whose depth fusion source is "depth" it
comes from the sensor's palm plane (docs/results/m6_wrist_flip.md); otherwise it was PnP's,
which flips. Clip `ego_depth_424` through the tracker (fusion v3, 98 % depth frames),
`tools/analyze_wrist_rotation.py`:

- step between frames, in 5 s windows of the first 25 s: p95 0.2 to 2.0 degrees;
- in the last 10 s (the "free" phase) there are bursts: 9 to 14 steps over 20 degrees, most of
  them within one second, on both hands at once, about the viewing axis, on depth frames.
  Their cause is not established.

## What was added

| piece | what it does |
|---|---|
| `vision/rotation_hold.py`, `demo/demo_tracker_node_v3.py` | repeats the last sensor rotation for up to 30 frames when a frame has none, instead of PnP's |
| `arm/arm_follow_node_v7.py --max-rotation D` | the EEF target turns at most D degrees from the orientation the node started with |
| the same option, late engage | after a (re-)engage the EEF keeps its orientation until the wrist has turned less than 5 degrees per frame for 10 frames, then latches the rotation reference |
| `arm/gate.py` (unchanged) | with rotation followed, a step over 10 rad/s is rejected; after 0.3 s of rejections the arm re-engages without a jump |
| `config/teleop_center_rot.yaml` | teleop_center.yaml with `position_only: false`, `max_rotation_deg: 45` |
| `tools/record_run_v3.py`, `tools/analyze_wrist_rotation.py` | the run's log now has the wrist orientation; the run script prints its steadiness |

## Simulated replay (`ego_depth_424`, arms from the centred start, 1.0 rad/s, domain 77)

Before the late engage the rotation reference was latched on the hand's first pose, the EEF
went to the 45 degree limit and stayed there. With it:

| | left | right |
|---|---|---|
| EEF turned from the start, p50 / max | 33 / 39 deg | 13 / 39 deg |
| time at the limit | 0 % | 0 % |
| EEF rotation per hand rotation, about forward / left / up | 0.09 / 0.28 / 0.14 | 0.42 / 0.28 / 0.77 |
| correlation of the two | 0.21 / 0.56 / 0.16 | 0.51 / 0.38 / 0.80 |
| joints at the 0.9 rad box | joint 4, 5, 7 | joint 7 |

So the arm does turn with the hand, but far from 1:1. Known reasons: the 0.9 rad joint box
(the wrist joints sit on it), and the IK trading rotation against position (rotation weight
0.8, one speed cap for all joints). The clip also has little deliberate rotation, so these
ratios are a weak measurement.

## Next

1. A recording with deliberate turns, with the camera where it is now: palm up / palm down,
   wrist up / down, hand left / right, each slowly, three times.
   `tools/run_teleop_real_v2.sh r1 --arms sim --config config/teleop_center_rot.yaml` moves
   nothing and shows the simulated arms in RViz.
2. From that: a joint box per joint (the wrist joints need more than 0.9 rad for a palm flip;
   the bridge takes one number today), and the IK's rotation weight.
3. Whether the robot's hand at the start pose points the way the operator's hand does. The
   mapping is relative: the EEF turns about the same world axes as the hand, starting from
   whatever orientation the robot's hand has.

## Run r1 (2026-11-26): deliberate turns, arms simulated

The camera where it is now, hands 0.33 to 0.35 m away, depth on 90 to 100 % of the wrist poses.

| | left | right |
|---|---|---|
| frames with the hand found | 568 of 1646 (35 %) | 709 of 1646 (43 %) |
| gaps over 0.3 s | 5 (0.8 to 2.1 s) | 3 (0.3 to 1.9 s) |
| orientation step between frames, p50 / p95 | 0.9 / 16.0 deg | 1.1 / 49.8 deg |
| steps over 20 degrees | 17 | 67 |
| wrist samples rejected by the gate | 47 | 107 |

Across the big steps the finger direction stays (dot product of the palm frame's finger axis
before and after: 0.98, 0.99) while the palm normal and the side axis turn by 50 to 60
degrees. So the jumps are a roll about the fingers: the sensor's palm plane does not give the
roll reliably while the hand is being flipped, and that is also when the hand is lost.
Where the fingers point (bending the wrist, turning the hand left and right) is steady.

The run's log has no images or landmarks, so other estimates of the roll cannot be tried on
it. `tools/record_ego_turns.py` records a clip with flip, bend, turn and grasp phases for that.

## Clip ego_turns (2026-11-26): four estimates compared, and what spoiled the comparison

`tools/eval_wrist_rotation.py data/clips/ego_turns` (1496 frames; pnp, sensor plane, MediaPipe
world landmarks, and a hybrid: finger direction from the sensor, roll from the world landmarks).
Step between consecutive frames, p95 in degrees / steps over 20 degrees:

| phase | hand found L / R | wrist's height in the image (0 top, 1 bottom) | pnp | sensor | world | hybrid |
|---|---|---|---|---|---|---|
| still | 20 / 45 % | 0.84 / 0.74 | 58.7 / 9, 26.3 / 7 | 3.6 / 1, 5.4 / 1 | 26.4 / 2, 17.0 / 2 | 31.3 / 3, 17.5 / 2 |
| flip | 89 / 99 % | 0.81 / 0.74 | 36.6 / 24, 77.9 / 56 | 6.9 / 5, 11.7 / 13 | 12.1 / 7, 16.9 / 14 | 14.3 / 5, 19.7 / 17 |
| bend | 98 / 98 % | 0.69 / 0.69 | 39.8 / 22, 68.7 / 68 | 18.3 / 14, 148.5 / 26 | 21.7 / 17, 58.8 / 36 | 46.9 / 23, 57.8 / 36 |
| turn | 100 / 100 % | 0.96 / 0.98 | 39.8 / 37, 70.6 / 52 | 30.5 / 47, 31.0 / 35 | 6.7 / 1, 16.3 / 9 | 24.2 / 41, 24.6 / 24 |
| grasp | 100 / 100 % | 0.92 / 0.96 | 89.3 / 55, 77.2 / 72 | 91.1 / 40, 19.7 / 13 | 66.3 / 43, 49.3 / 43 | 65.7 / 49, 57.4 / 41 |

(each cell: left, right.)

- The hands are low in the picture throughout, and in the turn and grasp phases the wrist is
  on the bottom edge (0.92 to 0.98 of the height; the contact sheet shows the wrists cut off).
  The sensor plane needs the palm's pixels, so there it is the worst (finger direction
  stepping about 20 degrees); the world landmarks hold up better there.
- Where the hands are further up (still, flip) the sensor plane is the steadiest: p95 3.6 to
  11.7 degrees, and PnP is the worst everywhere.
- All four share steps of 150 to 180 degrees on the same frames. A continuity rule (take the
  half turn about the fingers if it is at least 30 degrees closer to the last frame) removes
  some (right hand, bend, sensor: p95 148.5 -> 24.0; left, grasp: 91.1 -> 29.1) and not others.
- In the turn phase the palms faced the camera, not down, so the phases are not the motions
  the tool asked for. The clip cannot rank the estimates for a palm-down turn.

Conclusion: no estimate is good enough for 1:1 rotation on this clip, and the clip itself is
limited by where the hands are in the picture. The camera has to look further down (the
wrists at the picture's middle would need about 25 degrees more pitch for hands held where
they were) before the estimates can be judged.
