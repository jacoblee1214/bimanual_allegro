#!/usr/bin/env python3
"""
demo_v2.launch.py — the demo: a camera sees a person's two hands, two simulated xArm7 arms
with Allegro hands follow them in RViz, and with real:=true the two real arms follow too.

v2: the vision runs demo/demo_tracker_node_v2.py with depth fusion v3
(docs/results/m7_hold_v3.md) instead of v2.

Start it with demo/run_demo_v2.sh, which picks the camera and asks before real arms move
(docs/07_demo.md). Per side, as bimanual_sim_v2.launch.py, but with today's nodes:

    demo/demo_tracker_node_v2.py (vision node v5 + /demo/image)
      /<side>/wrist_pose -> arm/arm_follow_node_v6.py -> /<side>/arm/joint_states -> RViz
                                                       -> arm/xarm_bridge_node_v3.py --enable (real:=true)
      /<side>/allegro/vision/world_landmarks -> retargeting -> simulated Allegro hand

The real Allegro hands are not driven. Follower and bridge settings are in config/demo.yaml:
real arms get R5/R6's (radius 7 cm, 0.3 rad/s, joints within 0.3 rad), the simulation looser
ones. With real:=true the followers start from the arms' measured pose, and when the bridges
stop (after `duration` s) everything stops. real:=dry
runs the same with the bridges in dry run, to rehearse without moving anything.

    ros2 launch launch/demo_v2.launch.py source:=realsense view:=ego depth:=true
    ros2 launch launch/demo_v2.launch.py source:=0 view:=front

Arguments:
  source:=realsense | <camera index>    view:=ego | front    depth:=false
  real:=false | dry | true (dry: bridges read the arms and send nothing)
  duration:=60 (s, real:=dry/true only)    gpu:=true    rviz:=true
  hand_repo:=~/allegro_vision_teleop
"""
import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

REPO = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
BANNERS = {"false": "SIMULATION", "dry": "REAL ARMS: DRY RUN (not moving)", "true": "REAL ARMS MOVING"}


def _setup(context):
    arg = context.launch_configurations
    real = arg["real"] in ("true", "dry")
    ego = arg["view"] == "ego"
    hand_repo = Path(os.path.expanduser(arg["hand_repo"]))
    hand_python = str(hand_repo / ".venv" / "bin" / "python")
    retarget_python = os.environ.get("RETARGET_PY", str(hand_repo / ".venv-retarget" / "bin" / "python"))
    arm_python = str(REPO / ".venv-arm" / "bin" / "python")
    vision_python = str(REPO / ".venv-vision" / "bin" / "python")
    layout = yaml.safe_load((REPO / "config" / "arms_sim.yaml").read_text())
    demo = yaml.safe_load((REPO / "config" / "demo.yaml").read_text())
    arm_urdf = (REPO / "data" / "models" / "xarm7.urdf").read_text()
    follow = demo["follow"]["real" if real else "sim"]

    actions, bridges = [], []
    for side in SIDES:
        base = layout["arms"][side]
        x, y, z = (str(v) for v in base["base_xyz"])
        roll, pitch, yaw = (str(v) for v in base["base_rpy"])
        follow_cmd = [arm_python, str(REPO / "arm" / "arm_follow_node_v6.py"), "--position-only",
                      "--max-joint-speed", str(follow["max_joint_speed"]), "--radius", str(follow["radius"]),
                      "--scale", str(follow["scale"]), "--joint-box", str(follow["joint_box"])]
        follow_cmd += ["--q0-from-robot"] if real else ["--q0", *(str(v) for v in demo["sim_q0"][side])]
        if ego:
            follow_cmd += ["--camera", str(REPO / "config" / "ego_camera.yaml")]
        actions += [
            Node(package="tf2_ros", executable="static_transform_publisher", name=f"world_to_{side}_arm",
                 output="log",
                 arguments=["--x", x, "--y", y, "--z", z, "--roll", roll, "--pitch", pitch, "--yaw", yaw,
                            "--frame-id", "world", "--child-frame-id", f"{side}/arm/world"]),
            Node(package="robot_state_publisher", executable="robot_state_publisher",
                 namespace=f"{side}/arm", output="log",
                 parameters=[{"robot_description": arm_urdf, "frame_prefix": f"{side}/arm/"}]),
            ExecuteProcess(name=f"arm_follow_{side}", cmd=[*follow_cmd, "--ros-args", "-r", f"__ns:=/{side}"],
                           output="screen"),
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
            ExecuteProcess(name=f"hand_sim_bridge_{side}",
                           cmd=[hand_python, str(hand_repo / "sim_bridge_node.py"), "--mode", "sim", "--hand", side,
                                "--ros-args", "-r", f"__ns:=/{side}", "--log-level", "warn"],
                           output="screen"),
        ]
        if real:
            bridges.append(ExecuteProcess(
                name=f"xarm_bridge_{side}",
                cmd=[arm_python, str(REPO / "arm" / "xarm_bridge_node_v3.py"), "--side", side,
                     *(["--enable"] if arg["real"] == "true" else []), "--duration", arg["duration"], "--max-joint-speed", str(demo["bridge"]["max_joint_speed"]),
                     "--max-excursion", str(demo["bridge"]["max_excursion"]), "--ros-args", "-r", f"__ns:=/{side}"],
                output="screen"))
    actions += bridges
    for bridge in bridges:
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=bridge, on_exit=[EmitEvent(event=Shutdown(reason=f"{bridge.name} stopped"))])))

    vision = [vision_python, str(REPO / "demo" / "demo_tracker_node_v2.py"), "--source", arg["source"],
              "--view", arg["view"], "--banner", BANNERS[arg["real"]]]
    if arg["source"] != "realsense":
        vision += ["--camera-info", str(REPO / "config" / "camera_nominal_640x480.yaml")]
    if ego:
        vision += ["--depth", "--fusion", "v3"] if arg["depth"] == "true" else ["--ray-filter"]
    if arg["gpu"] == "true":
        vision.append("--gpu")
    actions.append(ExecuteProcess(name="hand_tracker", cmd=vision, output="screen"))

    if arg["rviz"] == "true":
        actions.append(Node(package="rviz2", executable="rviz2", output="log",
                            arguments=["-d", str(REPO / "config" / "demo.rviz")]))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="realsense", description="'realsense' or a camera index"),
        DeclareLaunchArgument("view", default_value="ego", description="ego (camera above and behind) or front"),
        DeclareLaunchArgument("depth", default_value="false", description="with view:=ego: the RealSense's depth"),
        DeclareLaunchArgument("real", default_value="false",
                              description="true: also move the real arms; dry: connect to them, send nothing"),
        DeclareLaunchArgument("duration", default_value="60", description="s; with real:=true the bridges stop then"),
        DeclareLaunchArgument("gpu", default_value="true", description="MediaPipe on the NVIDIA GPU"),
        DeclareLaunchArgument("rviz", default_value="true", description="open RViz"),
        DeclareLaunchArgument("hand_repo",
                              default_value=os.environ.get("ALLEGRO_TELEOP_DIR", "~/allegro_vision_teleop"),
                              description="the allegro_vision_teleop checkout"),
        OpaqueFunction(function=_setup),
    ])
