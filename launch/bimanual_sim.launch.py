#!/usr/bin/env python3
"""
bimanual_sim.launch.py — both Allegro Hands in simulation, driven by the vision node.

One camera (or a recorded clip) -> vision/hand_tracker_node.py -> per side, the hand
repository's single-hand pipeline under its own namespace:

    /<side>/allegro/vision/world_landmarks -> retargeting_node_v3.py -> /<side>/allegro/target_joints
    -> sim_bridge_node.py -> /<side>/allegro_hand_position_controller/commands
    -> ros2_control mock hand (allegro_hand_bringup.launch.py) -> /<side>/joint_states

Nothing here is a copy of the hand repository: its launch file and nodes are started
as they are, with `namespace`, `frame_prefix` and `--ros-args -r __ns:=/<side>`. TF frames
are prefixed `left/`, `right/`, and this file places the two hand bases in `world`.

Needs ROS 2 Jazzy and the Wonik driver workspace sourced, as the hand repository does:
    source /opt/ros/jazzy/setup.bash
    source ~/v6f_manuse_teleoperation/install/setup.bash
    ros2 launch launch/bimanual_sim.launch.py source:=data/clips/v2_cross.avi loop:=true gpu:=true

Arguments:
  source:=0                 camera index or video file for the vision node
  loop:=false               replay a clip over and over
  gpu:=false                MediaPipe on the NVIDIA GPU (docs/01_vision_design.md, V5)
  rviz:=true                open RViz with both hands and the camera image
  hand_repo:=$ALLEGRO_TELEOP_DIR or ~/allegro_vision_teleop
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch_ros.actions import Node

REPO = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")

# Where each hand base sits in `world`. The hand repo stands a single hand upright at the
# origin (fingers up, see its launch file); two of them go side by side, the left hand on
# +Y as on a robot facing +X.
BASE_Y = {"left": 0.12, "right": -0.12}


def _setup(context):
    arg = context.launch_configurations
    hand_repo = Path(os.path.expanduser(arg["hand_repo"]))
    # The hand repo's interpreters: .venv has mediapipe (numpy 1.x) and is the one this
    # repository's tests use too; .venv-retarget has dex_retargeting + pinocchio (numpy 2.x).
    python = str(hand_repo / ".venv" / "bin" / "python")
    retarget_python = os.environ.get("RETARGET_PY", str(hand_repo / ".venv-retarget" / "bin" / "python"))

    actions = []
    for side in SIDES:
        actions.append(IncludeLaunchDescription(
            str(hand_repo / "launch" / "allegro_hand_bringup.launch.py"),
            launch_arguments={
                "hardware": "mock_components",
                "hand": side,
                "rviz": "false",
                "namespace": side,
                "frame_prefix": f"{side}/",
                "world_tf": "false",
                # The bridge's URDF-coordinate copy, as the hand repo's sim mode reads it.
                "joint_states_topic": "allegro/joint_states_urdf",
            }.items(),
        ))
        actions.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"world_to_{side}_base",
            output="log",
            arguments=["--y", str(BASE_Y[side]), "--frame-id", "world", "--child-frame-id", f"{side}/base_link"],
        ))
        actions.append(ExecuteProcess(
            name=f"retarget_{side}",
            cmd=[retarget_python, str(hand_repo / "retargeting_node_v3.py"),
                 "--hand", side, "--no-undo-mirror", "--quiet",
                 "--ros-args", "-r", f"__ns:=/{side}"],
            output="screen",
        ))
        actions.append(ExecuteProcess(
            name=f"bridge_{side}",
            cmd=[python, str(hand_repo / "sim_bridge_node.py"), "--mode", "sim", "--hand", side,
                 "--ros-args", "-r", f"__ns:=/{side}", "--log-level", "warn"],
            output="screen",
        ))

    vision = [python, str(REPO / "vision" / "hand_tracker_node.py"), "--source", arg["source"]]
    if arg["loop"] == "true":
        vision.append("--loop")
    if arg["gpu"] == "true":
        vision.append("--gpu")
    actions.append(ExecuteProcess(name="hand_tracker", cmd=vision, output="screen"))

    if arg["rviz"] == "true":
        actions.append(Node(
            package="rviz2",
            executable="rviz2",
            output="log",
            arguments=["-d", str(REPO / "config" / "bimanual_sim.rviz")],
        ))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="0", description="camera index or video file"),
        DeclareLaunchArgument("loop", default_value="false", description="replay a clip over and over"),
        DeclareLaunchArgument("gpu", default_value="false", description="MediaPipe on the NVIDIA GPU"),
        DeclareLaunchArgument("rviz", default_value="true", description="open RViz"),
        DeclareLaunchArgument("hand_repo",
                              default_value=os.environ.get("ALLEGRO_TELEOP_DIR", "~/allegro_vision_teleop"),
                              description="the allegro_vision_teleop checkout"),
        OpaqueFunction(function=_setup),
    ])
