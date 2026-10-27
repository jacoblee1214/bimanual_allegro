#!/usr/bin/env bash
# run_demo.sh — start the demo (docs/07_demo.md): camera -> both hands -> simulated arms
# and hands in RViz; with --real the two real xArm7 arms follow too.
#
#   demo/run_demo.sh                       simulation only, camera picked automatically
#   demo/run_demo.sh --camera 0            a webcam (/dev/video0), operator facing it
#   demo/run_demo.sh --real                also the real arms, for 60 s (asks first)
#   demo/run_demo.sh --real --duration 30 --yes
#   demo/run_demo.sh --dry                 as --real, but the arms are only read, never moved
#
# Options: --camera realsense|<N>  --no-depth  --real | --dry  --duration S  --yes (do not ask)
# Ctrl-C stops everything. With --real the arms also stop by themselves after --duration.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

CAMERA=auto DEPTH=true REAL=false DURATION=60 YES=false
while [ $# -gt 0 ]; do
  case "$1" in
    --camera) CAMERA="$2"; shift ;;
    --no-depth) DEPTH=false ;;
    --real) REAL=true ;;
    --dry) REAL=dry ;;
    --duration) DURATION="$2"; shift ;;
    --yes) YES=true ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "[!] unknown option $1"; exit 2 ;;
  esac
  shift
done

# The same environment as the real-arm runs: ROS 2 Jazzy, the hand driver, the xArm meshes.
unset CONDA_EXE _CONDA_EXE _CONDA_ROOT CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
set +u
source /opt/ros/jazzy/setup.bash
source ~/v6f_manuse_teleoperation/install/setup.bash
source ~/xarm_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

if [ "$CAMERA" = auto ]; then
  FOUND="$("$REPO/.venv-vision/bin/python" "$REPO/demo/detect_camera.py")"
  case "$FOUND" in
    "realsense depth") CAMERA=realsense ;;
    realsense) CAMERA=realsense DEPTH=false ;;
    none) echo "[!] no camera found"; exit 1 ;;
    *) CAMERA="$FOUND" ;;
  esac
fi
if [ "$CAMERA" = realsense ]; then VIEW=ego; else VIEW=front DEPTH=false; fi
echo "camera: $CAMERA ($VIEW view, depth $DEPTH)"

if [ "$REAL" != false ]; then
  # Both arms must be connected, error-free and away from singularities before anything moves.
  POSE="$("$REPO/.venv-arm/bin/python" "$REPO/tools/check_pose_v2.py" 2>&1)" || true
  echo "$POSE" | grep -E "^(left|right):|-> " || true
  if [ "$(echo "$POSE" | grep -c -- "-> READY")" != 2 ]; then
    echo "[!] both arms must read READY (tools/check_pose_v2.py); not moving them"
    exit 1
  fi
fi
if [ "$REAL" = true ]; then
  echo
  echo "  REAL ARMS WILL MOVE for $DURATION s (radius 7 cm, 0.3 rad/s, joints within 0.3 rad)."
  echo "  Keep the emergency stop in reach. Hold both hands still for 2-3 s at the start."
  if [ "$YES" != true ]; then
    read -r -p "  Type 'yes' to start: " ANSWER
    [ "$ANSWER" = yes ] || { echo "not started"; exit 1; }
  fi
fi

exec ros2 launch "$REPO/launch/demo.launch.py" source:="$CAMERA" view:="$VIEW" depth:="$DEPTH" \
  real:="$REAL" duration:="$DURATION"
