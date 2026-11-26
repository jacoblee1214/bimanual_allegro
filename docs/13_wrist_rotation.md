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
