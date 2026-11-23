#!/usr/bin/env bash
# run_real_ego.sh — one real-arm run in the ego view, R5-R7 conditions: the ego RealSense
# sees both hands, both xArm7 arms follow (radius 7 cm, x0.5, 0.3 rad/s, joints within 0.3 rad).
#
#   tools/run_real_ego.sh <tag>              dry run: the bridges read the arms, send nothing
#   tools/run_real_ego.sh <tag> --enable     the arms move (asks for 'yes' first)
#
# Options: --duration S (32)  --fusion v2|v3 (v3, vision node v5)
# Both arms must read READY (tools/check_pose_v2.py). Logs go to data/runs/<tag>/:
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
if [ "$(echo "$POSE" | grep -c -- "-> READY")" != 2 ]; then
  echo "[!] both arms must read READY; not starting"; exit 1
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

start vision "$VISION_PY" vision/hand_tracker_node_v5.py --source realsense --view ego --depth --fusion "$FUSION" --gpu
"$ARM_PY" tools/record_run.py > "$LOG/record.csv" 2> "$LOG/record.log" & PIDS+=($!)
echo "camera starting..."; sleep 4
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
