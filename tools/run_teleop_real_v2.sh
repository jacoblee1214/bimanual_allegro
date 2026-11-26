#!/usr/bin/env bash
# run_teleop_real_v2.sh — run_teleop_real.sh plus --show rps (docs/11_ces_show_rps.md). Ego camera -> two xArm7 arms + two Allegro hands, one step at a time
# (docs/10_real_teleop_arms_hands.md). Checks the camera, the arms and the hands, measures the
# camera tilt, asks before anything real moves, records, and prints a summary.
#
#   tools/run_teleop_real_v2.sh <tag>                              arms dry run, hands simulated
#   tools/run_teleop_real_v2.sh <tag> --show rps                   the same, with rock-paper-scissors
#   tools/run_teleop_real_v2.sh <tag> --arms real --hands both     everything
#
# Options: --arms sim|dry|real (dry)   --hands mock|left|right|both (mock)
#          --duration S (60)           --no-rviz
#          --config FILE (config/teleop_real.yaml; teleop_same_15cm.yaml: 1:1 within 15 cm;
#                         teleop_center.yaml: hands 0.4 m apart, 1:1 within 20 cm;
#                         teleop_center_rot.yaml: the same, and the wrist's rotation is followed)
#          --arm-speed RAD_PER_S (the config's)
#          --show none|rps (none)      --rps-mode fair|reader (fair)   V sign 1 s: play; keys G, M
# Arms: the limits come from the config and are printed before the start;
# they start from READY or from the home pose; if they are elsewhere the script offers to bring
# them home (tools/home_arms.py). Hands: 3 rad/s per joint. Ctrl-C stops everything.
# Logs: data/runs/<tag>/ (launch.log, record.csv, ego_camera.yaml, check_*.txt).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

[ $# -ge 1 ] && [[ "$1" != -* ]] || { sed -n 2,20p "$0"; exit 2; }
TAG="$1"; shift
ARMS=dry HANDS=mock DURATION=60 RVIZ=true SHOW=none RPS_MODE=fair CONFIG=config/teleop_real.yaml ARM_SPEED=
while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="$2"; shift ;;
    --hands) HANDS="$2"; shift ;;
    --duration) DURATION="$2"; shift ;;
    --no-rviz) RVIZ=false ;;
    --show) SHOW="$2"; shift ;;
    --config) CONFIG="$2"; shift ;;
    --arm-speed) ARM_SPEED="$2"; shift ;;
    --rps-mode) RPS_MODE="$2"; shift ;;
    *) echo "[!] unknown option $1"; exit 2 ;;
  esac
  shift
done
case "$SHOW/$RPS_MODE" in none/*|rps/fair|rps/reader) ;; *) echo "[!] --show none|rps, --rps-mode fair|reader"; exit 2 ;; esac
case "$ARMS" in sim|dry|real) ;; *) echo "[!] --arms must be sim, dry or real"; exit 2 ;; esac
case "$HANDS" in mock) REAL_HANDS=() ;; left|right) REAL_HANDS=("$HANDS") ;; both) REAL_HANDS=(left right) ;;
  *) echo "[!] --hands must be mock, left, right or both"; exit 2 ;; esac

HAND_REPO="${ALLEGRO_TELEOP_DIR:-$HOME/allegro_vision_teleop}"
CONFIG="$(realpath "$CONFIG")"
LIMITS="$(python3 -c '
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); f, b = c["follow"], c["bridge"]
v = float(sys.argv[2]) if sys.argv[2] else b["max_joint_speed"]
print("radius %.0f cm, x%s, %s rad/s, joints within %s rad of the start" % (f["radius"] * 100, f["scale"], v, b["max_excursion"]))
' "$CONFIG" "$ARM_SPEED")"
unset CONDA_EXE _CONDA_EXE _CONDA_ROOT CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH VIRTUAL_ENV
set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/v6f_manuse_teleoperation/install/setup.bash"
[ -f "$HOME/xarm_ws/install/setup.bash" ] && source "$HOME/xarm_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
ARM_PY="$REPO/.venv-arm/bin/python" VISION_PY="$REPO/.venv-vision/bin/python"
LOG="$REPO/data/runs/$TAG"
[ ! -e "$LOG" ] || { echo "[!] $LOG exists; pick another tag"; exit 1; }
mkdir -p "$LOG"

echo "[1/4] camera"
USB="$("$VISION_PY" -c 'import pyrealsense2 as rs
d = rs.context().query_devices()
print(d[0].get_info(rs.camera_info.usb_type_descriptor) if len(d) else "none")')"
[[ "$USB" == 3* ]] || { echo "[!] RealSense on USB '$USB': it must be on USB 3 (cable and port)"; exit 1; }
"$VISION_PY" tools/measure_camera_tilt.py --out "$LOG/ego_camera.yaml" > "$LOG/camera_tilt.txt" 2>&1 \
  || { cat "$LOG/camera_tilt.txt"; echo "[!] could not measure the camera tilt"; exit 1; }
echo "  USB $USB; $(grep -E '^(pitch_down_deg|roll_deg)' "$LOG/ego_camera.yaml" | tr '\n' ' ')(R5-R7: pitch 30)"

echo "[2/4] arms: $ARMS ($LIMITS)"
if [ "$ARMS" != sim ]; then
  read_arms() {
    POSE="$("$ARM_PY" tools/check_pose_v2.py 2>&1)" || true
    echo "$POSE" > "$LOG/check_pose.txt"
    STARTS="$(echo "$POSE" | python3 -c '
import re, sys
import math, yaml
HOME = {s: [math.degrees(v) for v in q] for s, q in yaml.safe_load(open(sys.argv[1]))["home_q"].items()}
text = sys.stdin.read()
for side in ("left", "right"):
    m = re.search(side + r": state (\d+), error (\d+).*?joints deg\s+\[([^\]]*)\].*?-> (\S+)", text, re.S)
    if not m: print(" ", side, "unread"); continue
    q = [float(v) for v in m.group(3).split(",")]
    home = max(abs(a - b) for a, b in zip(q, HOME[side])) < 2.0
    where = "READY" if m.group(4) == "READY" else "HOME" if home else "no"
    print(" ", side, f"error {m.group(2)}", where if m.group(2) == "0" else "no")
' "$CONFIG")"
    echo "$STARTS"
  }
  read_arms
  if echo "$STARTS" | grep -qvE " (READY|HOME)$" && ! echo "$STARTS" | grep -qE "unread|error [1-9]"; then
    echo "  The arms are not at this config's start pose. Plan:"
    "$ARM_PY" tools/home_arms.py --config "$CONFIG" | sed "s/^/    /"
    read -r -p "  Type 'home' to move both arms there at 10 deg/s, left first: " ANSWER
    [ "$ANSWER" = home ] || { echo "not started"; exit 1; }
    "$ARM_PY" tools/home_arms.py --config "$CONFIG" --enable | grep -E "done|home|failed" | sed "s/^/    /"
    read_arms
  fi
  if echo "$STARTS" | grep -qvE " (READY|HOME)$"; then
    echo "[!] both arms must be at READY or at the config's start pose, with no error; not starting"; exit 1
  fi
fi

echo "[3/4] hands: $HANDS"
for SIDE in "${REAL_HANDS[@]}"; do
  IP="$(python3 -c "import yaml; print(yaml.safe_load(open('config/robots.local.yaml')).get('allegro', {}).get('$SIDE', ''))")"
  [ -n "$IP" ] || { echo "[!] no allegro.$SIDE in config/robots.local.yaml"; exit 1; }
  if ss -tn | grep -q "$IP:502 "; then echo "[!] $SIDE hand: something is already connected to it (one connection only)"; exit 1; fi
  CHECK="$(python3 "$HAND_REPO/tools/check_hand.py" "$IP" 2>&1)" || true
  echo "$CHECK" > "$LOG/check_hand_$SIDE.txt"
  echo "$CHECK" | grep -qi "$SIDE HAND" || { echo "[!] $SIDE hand did not answer as a $SIDE hand; see $LOG/check_hand_$SIDE.txt"; exit 1; }
  echo "  $SIDE ok: $(echo "$CHECK" | grep -m1 'Firmware' | sed 's/^ *• *//')"
done

if [ "$ARMS" = real ] || [ ${#REAL_HANDS[@]} -gt 0 ]; then
  echo
  [ "$ARMS" = real ] && echo "  REAL ARMS WILL MOVE for $DURATION s ($LIMITS)."
  [ ${#REAL_HANDS[@]} -gt 0 ] && echo "  REAL HAND(S) WILL MOVE: ${REAL_HANDS[*]}. The fingers straighten as the driver starts, then follow yours."
  echo "  Keep the emergency stop in reach and clear the space around the hands."
  read -r -p "  Type 'yes' to start: " ANSWER
  [ "$ANSWER" = yes ] || { echo "not started"; exit 1; }
fi

echo "[4/4] start: show both hands to the camera (logs in $LOG)"
"$ARM_PY" tools/record_run_v3.py > "$LOG/record.csv" 2> "$LOG/record.log" & REC=$!
# No wrist pose within 30 s: stop, instead of a run with nothing to follow (dry7b, 2026-11-23).
( for _ in $(seq 60); do sleep 0.5; grep -q "_wrist," "$LOG/record.csv" && exit 0; done
  echo "[!] no wrist pose in 30 s; stopping"; kill -INT "$(cat "$LOG/launch.pid")" ) & WATCH=$!
trap 'kill -TERM $REC $WATCH 2>/dev/null || true' EXIT
STATUS=0
bash -c 'echo $$ > "$1/launch.pid"; shift; exec ros2 launch "$@"' _ "$LOG" launch/teleop_real_v2.launch.py show:="$SHOW" rps_mode:="$RPS_MODE" config:="$CONFIG" ${ARM_SPEED:+arm_speed:="$ARM_SPEED"} \
  arms:="$([ "$ARMS" = real ] && echo true || echo "$ARMS")" hands:="$HANDS" camera:="$LOG/ego_camera.yaml" \
  duration:="$DURATION" rviz:="$RVIZ" hand_repo:="$HAND_REPO" 2>&1 | tee "$LOG/launch.log" | grep --line-buffered -vE "pal_statistics|FIFO RT" || STATUS=$?
kill -TERM $REC $WATCH 2>/dev/null || true
wait $REC 2>/dev/null || true
echo
echo "== summary ($LOG/record.csv)"
"$ARM_PY" tools/analyze_run.py "$LOG/record.csv" || true
echo "== wrist orientation"
"$ARM_PY" tools/analyze_wrist_rotation.py "$LOG/record.csv" || true
grep -hE "round (starts|over)" "$LOG/launch.log" | sed "s/^.*rps_show\]: /  /" || true
grep -hE "outside the box [1-9]|controller error|Traceback" "$LOG/launch.log" | tail -n 5 || true
if ss -tn | grep -q ":502 "; then echo "[!] a hand connection (port 502) is still open: check for a leftover ros2_control_node"; fi
