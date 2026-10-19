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

1단계 비전 레이어가 있고, 2단계 양손 텔레옵은 시뮬레이션까지 연결했습니다. 설계와 측정 기록은 [docs/01_vision_design.md](docs/01_vision_design.md), 선행 연구 조사는 [docs/survey_vision_hand_teleop.md](docs/survey_vision_hand_teleop.md) 에 있습니다.

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
