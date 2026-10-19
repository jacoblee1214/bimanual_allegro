# bimanual_allegro

xArm7 두 대와 Allegro Hand V6 좌우 한 쌍으로 구성된 양팔 로봇 시스템입니다.

최종 목표는 사람이 작업하는 영상을 VLM 이 이해하고, 이 로봇이 같은 작업을 수행하게 하는 것입니다.

손 한 쪽의 비전 텔레오퍼레이션은 [allegro_vision_teleop](https://github.com/jacoblee1214/allegro_vision_teleop) 에서 다룹니다. 이 저장소는 그 위에 양손, 팔, 데이터 기록, VLM 을 조립합니다.

## 진행 순서

1. 비전: 두 손 추적, 좌우 구분, 손목 자세 발행
2. 양손 텔레옵: `/left`, `/right` 네임스페이스로 손 파이프라인 두 벌
3. 팔 EEF 추종: 손목 자세에서 xArm7 IK 로
4. 팔과 손 결합, 데이터 기록
5. VLM

## 지금 있는 것

1단계 비전 레이어가 있고, 2단계 양손 텔레옵은 시뮬레이션까지 연결했습니다. 3단계 팔 추종은 양팔+양손 시뮬레이션까지 연결했고, 실물은 dry run 까지 확인했습니다([설계 v2](docs/03_arm_follow_design_v2.md), 결과 [M1](docs/results/m1_arm_ik.md), [M2 미분 IK](docs/results/m2_diff_ik.md), [M2 손 크기 배율 실험](docs/results/m2_scale_jitter.md)). R&D 후보 조사는 [docs/04_rnd_candidates_survey.md](docs/04_rnd_candidates_survey.md), 에이전트 작업 규칙은 [AGENTS.md](AGENTS.md) 에 있습니다. 설계와 측정 기록은 [docs/01_vision_design.md](docs/01_vision_design.md), 선행 연구 조사는 [docs/survey_vision_hand_teleop.md](docs/survey_vision_hand_teleop.md) 에 있습니다.

```bash
# 환경: ROS 2 Jazzy, --system-site-packages 로 만든 venv 에 requirements.txt
source /opt/ros/jazzy/setup.bash
python -m pytest                                        # 하드웨어 없이 돌아갑니다

tools/calibrate_camera.py                               # 카메라 보정 (체커보드)
vision/hand_tracker_node.py --source 0 --camera-info config/camera_<name>.yaml --gpu
```

`--gpu` 는 MediaPipe 의 Tasks API 를 GPU 에서 돌립니다. 두 손 기준으로 한 프레임에 4 ms 가 걸리고, 빼면 CPU 에서 30 ms 가 걸립니다(설계안 V5). 모델 파일은 git 에 없으므로 한 번 받아 둡니다.

```bash
mkdir -p data/models && curl -fL -o data/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
sha256sum data/models/hand_landmarker.task
# fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

노트북처럼 화면이 내장 그래픽에 연결된 PC 에서는, 따로 지정하지 않으면 MediaPipe 가 내장 그래픽을 잡아 오히려 CPU 보다 5배 느려집니다. `--gpu` 를 쓰면 노드가 NVIDIA GPU 를 쓰도록 환경 변수를 알아서 설정합니다.

### 양손 시뮬레이션 (2단계)

로봇 없이 두 손을 RViz 에서 움직여 봅니다. 손 저장소([allegro_vision_teleop](https://github.com/jacoblee1214/allegro_vision_teleop))와 원익 드라이버 워크스페이스가 필요합니다. 구성과 확인 결과는 [docs/02_bimanual_hand_sim.md](docs/02_bimanual_hand_sim.md) 에 있습니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/v6f_manuse_teleoperation/install/setup.bash
ros2 launch launch/bimanual_sim.launch.py source:=0 gpu:=true          # 카메라
ros2 launch launch/bimanual_sim.launch.py source:=data/clips/v2_cross.avi loop:=true gpu:=true
```

손 저장소가 다른 곳에 있으면 `hand_repo:=<경로>` 나 `ALLEGRO_TELEOP_DIR` 로 알려 줍니다.

### 팔 (3단계)

팔 코드는 pinocchio 와 xArm SDK 를 쓰는 별도 환경에서 돕니다. 비전 환경과 numpy 버전이 달라서입니다.

```bash
python3 -m venv --system-site-packages .venv-arm
.venv-arm/bin/pip install pin==4.1.0 xarm-python-sdk==1.18.4 "scipy>=1.14"
# xArm7 URDF: xarm_ros2 (jazzy) 의 xarm_description 을 빌드한 뒤
xacro $(ros2 pkg prefix xarm_description)/share/xarm_description/urdf/xarm_device.urdf.xacro dof:=7 robot_type:=xarm > data/models/xarm7.urdf
.venv-arm/bin/python tools/eval_arm_ik.py              # 녹화 궤적으로 IK 평가
.venv-arm/bin/python tools/check_xarm.py <ip>           # 팔 상태 읽기만 (움직이지 않음)
```

양팔+양손 시뮬레이션 (xArm 메시를 위해 `~/xarm_ws` 도 source):

```bash
source ~/xarm_ws/install/setup.bash
ros2 launch launch/bimanual_sim_v2.launch.py source:=data/clips/v2_palm2.avi loop:=true gpu:=true
```

실물 브리지(`arm/xarm_bridge_node.py`)는 `--enable` 없이는 명령을 보내지 않습니다. 실물 시험 계획은 설계 v2 4장에 있습니다.

로봇 IP 는 `config/robots.local.yaml` 에 둡니다. git 이 무시하는 파일입니다.
