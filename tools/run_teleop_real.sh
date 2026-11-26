#!/usr/bin/env bash
# run_teleop_real.sh — ego camera -> two xArm7 arms + two Allegro hands, one step at a time
# (docs/10_real_teleop_arms_hands.md). Checks the camera, the arms and the hands, measures the
# camera tilt, asks before anything real moves, records, and prints a summary.
#
#   tools/run_teleop_real.sh <tag>                              arms dry run, hands simulated
#   tools/run_teleop_real.sh <tag> --arms real                  the arms move
#   tools/run_teleop_real.sh <tag> --hands right                the real right hand moves
#   tools/run_teleop_real.sh <tag> --arms real --hands both     everything
#
# Options: --arms sim|dry|real (dry)   --hands mock|left|right|both (mock)
#          --duration S (60)           --no-rviz
# Arms: radius 7 cm, x0.5, 0.3 rad/s, joints within 0.3 rad of the start (R5's settings);
# they start from READY or from the home pose. Hands: 3 rad/s per joint. Ctrl-C stops everything.
# Logs: data/runs/<tag>/ (launch.log, record.csv, ego_camera.yaml, check_*.txt).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

[ $# -ge 1 ] && [[ "$1" != -* ]] || { sed -n 2,15p "$0"; exit 2; }
TAG="$1"; shift
ARMS=dry HANDS=mock DURATION=60 RVIZ=true
while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="$2"; shift ;;
    --hands) HANDS="$2"; shift ;;
    --duration) DURATION="$2"; shift ;;
    --no-rviz) RVIZ=false ;;
    *) echo "[!] unknown option $1"; exit 2 ;;
  esac
  shift
done
case "$ARMS" in sim|dry|real) ;; *) echo "[!] --arms must be sim, dry or real"; exit 2 ;; esac
case "$HANDS" in mock) REAL_HANDS=() ;; left|right) REAL_HANDS=("$HANDS") ;; both) REAL_HANDS=(left right) ;;
  *) echo "[!] --hands must be mock, left, right or both"; exit 2 ;; esac

HAND_REPO="${ALLEGRO_TELEOP_DIR:-$HOME/allegro_vision_teleop}"
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

echo "[2/4] arms: $ARMS"
if [ "$ARMS" != sim ]; then
  POSE="$("$ARM_PY" tools/check_pose_v2.py 2>&1)" || true
  echo "$POSE" > "$LOG/check_pose.txt"
  STARTS="$(echo "$POSE" | python3 -c '
import re, sys
HOME = {"left": [-0.4, 11.7, -0.1, 68.1, 180.0, 29.8, 0.0], "right": [-15.3, 12.2, 16.7, 68.0, 181.1, 30.0, 176.0]}
text = sys.stdin.read()
for side in ("left", "right"):
    m = re.search(side + r": state (\d+), error (\d+).*?joints deg\s+\[([^\]]*)\].*?-> (\S+)", text, re.S)
    if not m: print(" ", side, "unread"); continue
    q = [float(v) for v in m.group(3).split(",")]
    home = max(abs(a - b) for a, b in zip(q, HOME[side])) < 2.0
    where = "READY" if m.group(4) == "READY" else "HOME" if home else "no"
    print(" ", side, f"error {m.group(2)}", where if m.group(2) == "0" else "no")
')"
  echo "$STARTS"
  if echo "$STARTS" | grep -qvE " (READY|HOME)$"; then
    echo "[!] both arms must be at READY or at the home pose, with no error; not starting"; exit 1
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
  [ "$ARMS" = real ] && echo "  REAL ARMS WILL MOVE for $DURATION s (radius 7 cm, x0.5, 0.3 rad/s, joints within 0.3 rad)."
  [ ${#REAL_HANDS[@]} -gt 0 ] && echo "  REAL HAND(S) WILL MOVE: ${REAL_HANDS[*]}. The fingers straighten as the driver starts, then follow yours."
  echo "  Keep the emergency stop in reach and clear the space around the hands."
  read -r -p "  Type 'yes' to start: " ANSWER
  [ "$ANSWER" = yes ] || { echo "not started"; exit 1; }
fi

echo "[4/4] start: show both hands to the camera (logs in $LOG)"
"$ARM_PY" tools/record_run_v2.py > "$LOG/record.csv" 2> "$LOG/record.log" & REC=$!
# No wrist pose within 30 s: stop, instead of a run with nothing to follow (dry7b, 2026-11-23).
( for _ in $(seq 60); do sleep 0.5; grep -q "_wrist," "$LOG/record.csv" && exit 0; done
  echo "[!] no wrist pose in 30 s; stopping"; kill -INT "$(cat "$LOG/launch.pid")" ) & WATCH=$!
trap 'kill -TERM $REC $WATCH 2>/dev/null || true' EXIT
STATUS=0
bash -c 'echo $$ > "$1/launch.pid"; shift; exec ros2 launch "$@"' _ "$LOG" launch/teleop_real.launch.py \
  arms:="$([ "$ARMS" = real ] && echo true || echo "$ARMS")" hands:="$HANDS" camera:="$LOG/ego_camera.yaml" \
  duration:="$DURATION" rviz:="$RVIZ" hand_repo:="$HAND_REPO" 2>&1 | tee "$LOG/launch.log" | grep --line-buffered -vE "pal_statistics|FIFO RT" || STATUS=$?
kill -TERM $REC $WATCH 2>/dev/null || true
wait $REC 2>/dev/null || true
echo
echo "== summary ($LOG/record.csv)"
"$ARM_PY" tools/analyze_run.py "$LOG/record.csv" || true
grep -hE "outside the box [1-9]|controller error|Traceback" "$LOG/launch.log" | tail -n 5 || true
if ss -tn | grep -q ":502 "; then echo "[!] a hand connection (port 502) is still open: check for a leftover ros2_control_node"; fi
