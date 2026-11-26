#!/usr/bin/env bash
# run_real_ego_v3.sh — run_real_ego_v2.sh that does not start the arms without the camera; one real-arm run in the ego view, R5-R7 conditions: the ego RealSense
# sees both hands, both xArm7 arms follow (radius 7 cm, x0.5, 0.3 rad/s, joints within 0.3 rad).
#
#   tools/run_real_ego_v3.sh <tag>           dry run: the bridges read the arms, send nothing
#   tools/run_real_ego_v3.sh <tag> --enable  the arms move (asks for 'yes' first)
#
# Options: --duration S (32)  --fusion v2|v3|v4 (v3; v2, v3: vision node v5, v4: vision node v6)
# Both arms must read READY (tools/check_pose_v2.py) or sit at the home pose. Logs go to data/runs/<tag>/:
# one per node, and record.csv (tools/record_run.py). Ctrl-C stops everything; the
# bridges also stop the arms by themselves after --duration.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

[ $# -ge 1 ] && [[ "$1" != -* ]] || { sed -n 2,12p "$0"; exit 2; }
TAG="$1"; shift
ENABLE=false DURATION=32 FUSION=v3
while [ $# -gt 0 ]; do
  case "$1" in
    --enable) ENABLE=true ;;
    --duration) DURATION="$2"; shift ;;
    --fusion) FUSION="$2"; shift ;;
    *) echo "[!] unknown option $1"; exit 2 ;;
  esac
  shift
done

case "$FUSION" in
  v2|v3) NODE=vision/hand_tracker_node_v5.py ;;
  v4) NODE=vision/hand_tracker_node_v6.py ;;
  *) echo "[!] --fusion must be v2, v3 or v4"; exit 2 ;;
esac

unset CONDA_EXE _CONDA_EXE _CONDA_ROOT CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
ARM_PY="$REPO/.venv-arm/bin/python" VISION_PY="$REPO/.venv-vision/bin/python"
LOG="$REPO/data/runs/$TAG"
[ ! -e "$LOG" ] || { echo "[!] $LOG exists; pick another tag"; exit 1; }
mkdir -p "$LOG"

POSE="$("$ARM_PY" tools/check_pose_v2.py 2>&1)" || true
echo "$POSE" | tee "$LOG/check_pose.txt" | grep -E "^(left|right):|-> " || true
# Each arm starts from READY, or from the home pose (user's choice, 2026-11-24). Home fails check_pose_v2
# on |joint2| 12 deg, but offline the 7 cm sphere from home keeps sigma_min >= 0.13 (READY right: 0.116).
STARTS="$(echo "$POSE" | python3 -c '
import re, sys
HOME = {"left": [-0.4, 11.7, -0.1, 68.1, 180.0, 29.8, 0.0], "right": [-15.3, 12.2, 16.7, 68.0, 181.1, 30.0, 176.0]}
text = sys.stdin.read()
for side in ("left", "right"):
    m = re.search(side + r":.*?joints deg\s+\[([^\]]*)\].*?-> (\S+)", text, re.S)
    if not m: print(side, "unread"); continue
    q = [float(v) for v in m.group(1).split(",")]
    home = max(abs(a - b) for a, b in zip(q, HOME[side])) < 2.0
    print(side, "READY" if m.group(2) == "READY" else "HOME" if home else "no")
')"
echo "$STARTS"
if echo "$STARTS" | grep -qvE " (READY|HOME)$"; then
  echo "[!] both arms must be at READY or at the home pose; not starting"; exit 1
fi
EN=()
if [ "$ENABLE" = true ]; then
  echo
  echo "  REAL ARMS WILL MOVE for $DURATION s (radius 7 cm, x0.5, 0.3 rad/s, joints within 0.3 rad)."
  echo "  Keep the emergency stop in reach. Hold both hands still for 2-3 s at the start."
  read -r -p "  Type 'yes' to start: " ANSWER
  [ "$ANSWER" = yes ] || { echo "not started"; exit 1; }
  EN=(--enable)
fi

PIDS=()
start() { local name="$1"; shift; "$@" > "$LOG/$name.log" 2>&1 & PIDS+=($!); }
stop_all() {
  kill -TERM "${PIDS[@]}" 2>/dev/null || true
  sleep 2
  kill -KILL "${PIDS[@]}" 2>/dev/null || true
}
trap stop_all EXIT

start vision "$VISION_PY" "$NODE" --source realsense --view ego --depth --fusion "$FUSION" --gpu
"$ARM_PY" tools/record_run.py > "$LOG/record.csv" 2> "$LOG/record.log" & PIDS+=($!)
echo "camera starting... show both hands to the camera"
# dry7b (2026-11-23): the RealSense failed to start, the node died, and the run went on with 0 wrist samples.
for _ in $(seq 30); do
  kill -0 "${PIDS[0]}" 2>/dev/null || { echo "[!] vision node died:"; tail -n 3 "$LOG/vision.log"; exit 1; }
  grep -q "_wrist," "$LOG/record.csv" && break
  sleep 0.5
done
grep -q "_wrist," "$LOG/record.csv" || { echo "[!] no wrist pose in 15 s; not starting the arms"; exit 1; }
echo "camera ok: $(cut -d, -f1 "$LOG/record.csv" | grep _wrist | sort | uniq -c | tr -s ' \n' ' ')"
BRIDGES=()
for SIDE in left right; do
  start "follow_$SIDE" "$ARM_PY" arm/arm_follow_node_v6.py --q0-from-robot --position-only --max-joint-speed 0.3 \
    --radius 0.07 --scale 0.5 --joint-box 0.3 --camera config/ego_camera.yaml --ros-args -r "__ns:=/$SIDE"
  start "bridge_$SIDE" "$ARM_PY" arm/xarm_bridge_node_v3.py --side "$SIDE" "${EN[@]}" --duration "$DURATION" \
    --max-joint-speed 0.3 --max-excursion 0.3 --ros-args -r "__ns:=/$SIDE"
  BRIDGES+=("${PIDS[-1]}")
done
echo "running $DURATION s ($([ "$ENABLE" = true ] && echo MOVING || echo dry run)); logs in $LOG"
STATUS=0
for P in "${BRIDGES[@]}"; do wait "$P" || STATUS=$?; done
echo "bridges done (exit $STATUS)"
