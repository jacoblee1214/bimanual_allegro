#!/usr/bin/env python3
"""
bimanual_sim_v2.launch.py — two xArm7 arms with an Allegro Hand each, in simulation,
driven by the vision node: arms follow the wrists, hands follow the fingers.

v1 (bimanual_sim.launch.py) had the two hands standing on the floor. v2 puts each hand
on its arm's flange and adds, per side:

    /<side>/wrist_pose -> arm/arm_follow_node.py -> /<side>/arm/joint_states
    -> robot_state_publisher (xArm7 URDF, TF prefix <side>/arm/)

The fingers' path is v1's, unchanged. Bases sit where config/arms_sim.yaml says (1.5 m
apart, both facing +x); the flange -> hand transform there is a placeholder until the
mount is measured.

The vision node needs intrinsics for wrist poses. Without a calibration this uses
config/camera_nominal_640x480.yaml, a guess (fx = 600): wrist motion is then off by a
constant factor, which the clutch mapping turns into a speed scale, not a drift.

    source /opt/ros/jazzy/setup.bash
    source ~/v6f_manuse_teleoperation/install/setup.bash
    source ~/xarm_ws/install/setup.bash            # xArm meshes for RViz
    ros2 launch launch/bimanual_sim_v2.launch.py source:=data/clips/v2_palm2.avi loop:=true gpu:=true

Arguments: v1's (source, loop, gpu, rviz, hand_repo), plus
  camera_info:=config/camera_nominal_640x480.yaml
  max_joint_speed:=1.0        rad/s cap for the arm IK
"""
import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch_ros.actions import Node

REPO = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")


def _setup(context):
    arg = context.launch_configurations
    hand_repo = Path(os.path.expanduser(arg["hand_repo"]))
    python = str(hand_repo / ".venv" / "bin" / "python")
    retarget_python = os.environ.get("RETARGET_PY", str(hand_repo / ".venv-retarget" / "bin" / "python"))
    arm_python = str(REPO / ".venv-arm" / "bin" / "python")
    layout = yaml.safe_load((REPO / "config" / "arms_sim.yaml").read_text())
    arm_urdf = (REPO / "data" / "models" / "xarm7.urdf").read_text()

    actions = []
    for side in SIDES:
        base = layout["arms"][side]
        x, y, z = (str(v) for v in base["base_xyz"])
        roll, pitch, yaw = (str(v) for v in base["base_rpy"])
        actions += [
            # Arm: the URDF's root link is `world`, so with the prefix it is <side>/arm/world.
            Node(package="tf2_ros", executable="static_transform_publisher", name=f"world_to_{side}_arm",
                 output="log",
                 arguments=["--x", x, "--y", y, "--z", z, "--roll", roll, "--pitch", pitch, "--yaw", yaw,
                            "--frame-id", "world", "--child-frame-id", f"{side}/arm/world"]),
            Node(package="robot_state_publisher", executable="robot_state_publisher",
                 namespace=f"{side}/arm", output="log",
                 parameters=[{"robot_description": arm_urdf, "frame_prefix": f"{side}/arm/"}]),
            ExecuteProcess(name=f"arm_follow_{side}",
                           cmd=[arm_python, str(REPO / "arm" / "arm_follow_node.py"),
                                "--max-joint-speed", arg["max_joint_speed"],
                                "--ros-args", "-r", f"__ns:=/{side}"],
                           output="screen"),
            # Hand on the flange.
            Node(package="tf2_ros", executable="static_transform_publisher", name=f"{side}_flange_to_hand",
                 output="log",
                 arguments=[*sum(([f"--{k}", str(v)] for k, v in zip(("x", "y", "z"), layout["hand_mount_xyz"])), []),
                            *sum(([f"--{k}", str(v)] for k, v in zip(("roll", "pitch", "yaw"), layout["hand_mount_rpy"])), []),
                            "--frame-id", f"{side}/arm/link_eef", "--child-frame-id", f"{side}/base_link"]),
            IncludeLaunchDescription(
                str(hand_repo / "launch" / "allegro_hand_bringup.launch.py"),
                launch_arguments={
                    "hardware": "mock_components", "hand": side, "rviz": "false",
                    "namespace": side, "frame_prefix": f"{side}/", "world_tf": "false",
                    "joint_states_topic": "allegro/joint_states_urdf",
                }.items(),
            ),
            ExecuteProcess(name=f"retarget_{side}",
                           cmd=[retarget_python, str(hand_repo / "retargeting_node_v3.py"),
                                "--hand", side, "--no-undo-mirror", "--quiet",
                                "--ros-args", "-r", f"__ns:=/{side}"],
                           output="screen"),
            ExecuteProcess(name=f"bridge_{side}",
                           cmd=[python, str(hand_repo / "sim_bridge_node.py"), "--mode", "sim", "--hand", side,
                                "--ros-args", "-r", f"__ns:=/{side}", "--log-level", "warn"],
                           output="screen"),
        ]

    vision = [python, str(REPO / "vision" / "hand_tracker_node.py"), "--source", arg["source"],
              "--camera-info", str(REPO / arg["camera_info"])]
    if arg["loop"] == "true":
        vision.append("--loop")
    if arg["gpu"] == "true":
        vision.append("--gpu")
    actions.append(ExecuteProcess(name="hand_tracker", cmd=vision, output="screen"))

    if arg["rviz"] == "true":
        actions.append(Node(package="rviz2", executable="rviz2", output="log",
                            arguments=["-d", str(REPO / "config" / "bimanual_sim_v2.rviz")]))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="0", description="camera index or video file"),
        DeclareLaunchArgument("loop", default_value="false", description="replay a clip over and over"),
        DeclareLaunchArgument("gpu", default_value="false", description="MediaPipe on the NVIDIA GPU"),
        DeclareLaunchArgument("rviz", default_value="true", description="open RViz"),
        DeclareLaunchArgument("camera_info", default_value="config/camera_nominal_640x480.yaml",
                              description="intrinsics YAML, relative to the repository"),
        DeclareLaunchArgument("max_joint_speed", default_value="1.0", description="arm IK joint speed cap, rad/s"),
        DeclareLaunchArgument("hand_repo",
                              default_value=os.environ.get("ALLEGRO_TELEOP_DIR", "~/allegro_vision_teleop"),
                              description="the allegro_vision_teleop checkout"),
        OpaqueFunction(function=_setup),
    ])
