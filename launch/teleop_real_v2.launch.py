#!/usr/bin/env python3
"""
teleop_real_v2.launch.py — teleop_real.launch.py plus show:=rps. The ego camera sees the operator's two hands; two xArm7 arms follow
the wrists and two Allegro Hand V6 follow the fingers. Arms and hands are each simulated,
rehearsed or real, so the system can be brought up one step at a time
(docs/10_real_teleop_arms_hands.md). Start it with tools/run_teleop_real_v2.sh, which checks the
camera, the arms and the hands first and asks before anything real moves.

Per side, as demo_v2.launch.py:

    demo/demo_tracker_node_v2.py (vision node v5, depth fusion v3)
      /<side>/wrist_pose -> arm/arm_follow_node_v6.py -> /<side>/arm/joint_states -> RViz
                                                       -> arm/xarm_bridge_node_v3.py (arms:=dry|true)
      /<side>/allegro/vision/world_landmarks -> retargeting_node_v3.py -> hand/hand_bridge_node.py
                                              -> ros2_control: mock hand, or the real hand

Arguments:
  arms:=sim | dry | true     sim: no robot, start from the home pose (config/teleop_real.yaml)
                             dry: bridges read the real arms, send nothing;  true: arms move
  hands:=mock | left | right | both   which hands are the real ones (the others are simulated).
                             A real hand's driver straightens the fingers when it starts.
  source:=realsense | clip:<dir>      camera:=<ego camera YAML> (R_CB; default config/ego_camera.yaml)
  duration:=60 (s; everything stops then)    gpu:=true    rviz:=true
  hand_repo:=~/allegro_vision_teleop
  config:=config/teleop_real.yaml   speeds, radius, scale, joint box (config/teleop_same_15cm.yaml: 1:1)
  arm_speed:=<rad/s>         overrides the config's arm joint speed, follow node and bridge
  show:=none | rps           rps: show/rps_node.py sits between the teleop and the bridges and plays
                             rock-paper-scissors (docs/11_ces_show_rps.md);  rps_mode:=fair | reader
"""
import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

REPO = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
REAL_HANDS = {"mock": (), "left": ("left",), "right": ("right",), "both": SIDES}


def _setup(context):
    arg = context.launch_configurations
    if arg["arms"] not in ("sim", "dry", "true") or arg["hands"] not in REAL_HANDS:
        raise RuntimeError("arms must be sim, dry or true; hands must be mock, left, right or both")
    real_arms = arg["arms"] in ("dry", "true")
    real_hands = REAL_HANDS[arg["hands"]]
    hand_repo = Path(os.path.expanduser(arg["hand_repo"]))
    hand_python = str(hand_repo / ".venv" / "bin" / "python")
    retarget_python = os.environ.get("RETARGET_PY", str(hand_repo / ".venv-retarget" / "bin" / "python"))
    arm_python = str(REPO / ".venv-arm" / "bin" / "python")
    vision_python = str(REPO / ".venv-vision" / "bin" / "python")
    layout = yaml.safe_load((REPO / "config" / "arms_sim.yaml").read_text())
    cfg = yaml.safe_load(Path(os.path.expanduser(arg["config"])).read_text())
    if arg["arm_speed"]:
        cfg["follow"]["max_joint_speed"] = cfg["bridge"]["max_joint_speed"] = float(arg["arm_speed"])
    hand_ips = yaml.safe_load((REPO / "config" / "robots.local.yaml").read_text()).get("allegro", {}) if real_hands else {}
    arm_urdf = (REPO / "data" / "models" / "xarm7.urdf").read_text()
    follow = cfg["follow"]
    show = arg["show"] == "rps"
    if arg["show"] not in ("none", "rps"):
        raise RuntimeError("show must be none or rps")
    # With the show, the teleop's outputs go to show/rps_node.py, which passes them on or replaces them.
    arm_out = ["-r", "arm/joint_states:=arm/teleop_joint_states"] if show else []
    hand_out = ["-r", "allegro/target_joints:=allegro/teleop_target_joints"] if show else []

    actions, bridges = [], []
    for side in SIDES:
        base = layout["arms"][side]
        x, y, z = (str(v) for v in base["base_xyz"])
        roll, pitch, yaw = (str(v) for v in base["base_rpy"])
        hand_real = side in real_hands
        follow_cmd = [arm_python, str(REPO / "arm" / "arm_follow_node_v6.py"), "--position-only",
                      "--max-joint-speed", str(follow["max_joint_speed"]), "--radius", str(follow["radius"]),
                      "--scale", str(follow["scale"]), "--joint-box", str(follow["joint_box"]),
                      "--camera", arg["camera"]]
        follow_cmd += ["--q0-from-robot"] if real_arms else ["--q0", *(str(v) for v in cfg["home_q"][side])]
        actions += [
            Node(package="tf2_ros", executable="static_transform_publisher", name=f"world_to_{side}_arm",
                 output="log",
                 arguments=["--x", x, "--y", y, "--z", z, "--roll", roll, "--pitch", pitch, "--yaw", yaw,
                            "--frame-id", "world", "--child-frame-id", f"{side}/arm/world"]),
            Node(package="robot_state_publisher", executable="robot_state_publisher",
                 namespace=f"{side}/arm", output="log",
                 parameters=[{"robot_description": arm_urdf, "frame_prefix": f"{side}/arm/"}]),
            ExecuteProcess(name=f"arm_follow_{side}", cmd=[*follow_cmd, "--ros-args", "-r", f"__ns:=/{side}", *arm_out],
                           output="screen"),
            Node(package="tf2_ros", executable="static_transform_publisher", name=f"{side}_flange_to_hand",
                 output="log",
                 arguments=[*sum(([f"--{k}", str(v)] for k, v in zip(("x", "y", "z"), layout["hand_mount_xyz"])), []),
                            *sum(([f"--{k}", str(v)] for k, v in zip(("roll", "pitch", "yaw"), layout["hand_mount_rpy"])), []),
                            "--frame-id", f"{side}/arm/link_eef", "--child-frame-id", f"{side}/base_link"]),
            IncludeLaunchDescription(
                str(hand_repo / "launch" / "allegro_hand_bringup.launch.py"),
                launch_arguments={
                    "hardware": "hardware" if hand_real else "mock_components", "hand": side, "rviz": "false",
                    **({"descriptor": f"modbus_tcp:{hand_ips[side]}:502"} if hand_real else {}),
                    "namespace": side, "frame_prefix": f"{side}/", "world_tf": "false",
                    "joint_states_topic": "allegro/joint_states_urdf",
                }.items(),
            ),
            ExecuteProcess(name=f"retarget_{side}",
                           cmd=[retarget_python, str(hand_repo / "retargeting_node_v3.py"),
                                "--hand", side, "--no-undo-mirror", "--quiet",
                                "--ros-args", "-r", f"__ns:=/{side}", *hand_out],
                           output="screen"),
            ExecuteProcess(name=f"hand_bridge_{side}",
                           cmd=[hand_python, str(REPO / "hand" / "hand_bridge_node.py"),
                                "--mode", "real" if hand_real else "sim", "--hand", side,
                                "--alpha", str(cfg["hand"]["alpha"]),
                                "--max-joint-speed", str(cfg["hand"]["max_joint_speed"]),
                                "--hand-repo", str(hand_repo),
                                "--ros-args", "-r", f"__ns:=/{side}", "--log-level", "warn"],
                           output="screen"),
        ]
        if real_arms:
            bridges.append(ExecuteProcess(
                name=f"xarm_bridge_{side}",
                cmd=[arm_python, str(REPO / "arm" / "xarm_bridge_node_v3.py"), "--side", side,
                     *(["--enable"] if arg["arms"] == "true" else []), "--duration", arg["duration"],
                     "--max-joint-speed", str(cfg["bridge"]["max_joint_speed"]),
                     "--max-excursion", str(cfg["bridge"]["max_excursion"]), "--ros-args", "-r", f"__ns:=/{side}"],
                output="screen"))
    actions += bridges
    for bridge in bridges:
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=bridge, on_exit=[EmitEvent(event=Shutdown(reason=f"{bridge.name} stopped"))])))
    if not bridges:
        actions.append(TimerAction(period=float(arg["duration"]),
                                   actions=[EmitEvent(event=Shutdown(reason="duration over"))]))

    banner = (f"ARMS {dict(sim='SIM', dry='DRY RUN', true='MOVING')[arg['arms']]}"
              f" / HANDS {'REAL: ' + arg['hands'].upper() if real_hands else 'SIM'}")
    vision = [vision_python, str(REPO / "demo" / "demo_tracker_node_v2.py"), "--source", arg["source"],
              "--view", "ego", "--depth", "--fusion", "v3", "--banner", banner]
    if arg["gpu"] == "true":
        vision.append("--gpu")
    actions.append(ExecuteProcess(name="hand_tracker", cmd=vision, output="screen"))

    if show:
        actions.append(ExecuteProcess(
            name="rps_show",
            cmd=[vision_python, str(REPO / "show" / "rps_node.py"), "--mode", arg["rps_mode"],
                 "--joint-box", str(cfg["bridge"]["max_excursion"]),
                 *(["--auto-start", arg["rps_auto_start"]] if arg["rps_auto_start"] else []),
                 *(["--no-window"] if arg["rps_window"] != "true" else [])],
            output="screen"))

    if arg["rviz"] == "true":
        actions.append(Node(package="rviz2", executable="rviz2", output="log",
                            arguments=["-d", str(REPO / "config" / "demo.rviz")]))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("arms", default_value="sim", description="sim, dry (read the real arms) or true (move them)"),
        DeclareLaunchArgument("hands", default_value="mock", description="mock, left, right or both: the real hands"),
        DeclareLaunchArgument("source", default_value="realsense", description="'realsense' or clip:<dir>"),
        DeclareLaunchArgument("camera", default_value=str(REPO / "config" / "ego_camera.yaml"),
                              description="ego camera YAML with R_CB (tools/measure_camera_tilt.py)"),
        DeclareLaunchArgument("duration", default_value="60", description="s; everything stops then"),
        DeclareLaunchArgument("gpu", default_value="true", description="MediaPipe on the NVIDIA GPU"),
        DeclareLaunchArgument("rviz", default_value="true", description="open RViz"),
        DeclareLaunchArgument("config", default_value=str(REPO / "config" / "teleop_real.yaml"),
                              description="settings YAML (config/teleop_real.yaml, config/teleop_same_15cm.yaml)"),
        DeclareLaunchArgument("arm_speed", default_value="", description="rad/s; overrides the config"),
        DeclareLaunchArgument("show", default_value="none", description="none, or rps: rock-paper-scissors"),
        DeclareLaunchArgument("rps_mode", default_value="fair", description="fair or reader"),
        DeclareLaunchArgument("rps_window", default_value="true", description="the scoreboard window"),
        DeclareLaunchArgument("rps_auto_start", default_value="", description="s after the first hand (tests)"),
        DeclareLaunchArgument("hand_repo",
                              default_value=os.environ.get("ALLEGRO_TELEOP_DIR", "~/allegro_vision_teleop"),
                              description="the allegro_vision_teleop checkout"),
        OpaqueFunction(function=_setup),
    ])
