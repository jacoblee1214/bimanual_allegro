# 시연: 카메라 앞 사람의 두 손을 양팔 로봇이 따라가기

지금 수준(3단계, 팔 위치 추종)을 다른 사람에게 보여 주기 위한 구성입니다. 카메라를 꽂고 스크립트 하나를 실행하면 됩니다.

- 카메라가 두 손을 잡습니다. 화면에 손 골격과, 손마다 추적 상태(`left: tracking (depth)` 등)가 그려집니다.
- RViz 의 xArm7 두 대가 손목 위치를 따라가고, 그 끝의 Allegro 손 두 개가 손가락을 따라갑니다.
- `--real` 을 주면 실물 xArm7 두 대도 같이 따라갑니다. 실물 Allegro 손은 아직 움직이지 않습니다.

## 실행

```bash
demo/run_demo.sh                      # 시뮬레이션만. 카메라는 자동으로 고릅니다
demo/run_demo.sh --camera 0           # 웹캠 /dev/video0 (작업자가 카메라를 마주 봄)
demo/run_demo.sh --dry                # 실물 팔에 연결해 읽기만 합니다 (리허설)
demo/run_demo.sh --real               # 실물 팔도 60 s 동안 따라갑니다. 시작 전에 'yes' 를 묻습니다
demo/run_demo.sh --real --duration 30
```

Ctrl-C 로 모두 멈춥니다. 종료할 때 로그에 `KeyboardInterrupt` 와 `process has died` 가 찍히는데, ROS 2 노드들이 종료 신호를 받는 방식 때문이며 동작에는 영향이 없습니다.

## 카메라

`demo/detect_camera.py` 가 고릅니다.

| 연결된 카메라 | 시점 | 손목 거리 |
|---|---|---|
| RealSense, USB 3 | 에고(작업자 머리 위 뒤쪽, `config/ego_camera.yaml`) | 깊이 센서 (융합 v2, `docs/results/m6_wrist_flip.md`) |
| RealSense, USB 2 | 에고 | PnP + 시선 방향 칼만 필터 |
| 그 밖의 웹캠 | 정면(작업자가 카메라를 마주 봄) | PnP. 내부 파라미터는 `config/camera_nominal_640x480.yaml` 의 추정값(fx 600) |

- 웹캠은 `/dev/video` 순서로 처음 컬러 영상이 나오는 장치를 고릅니다. 적외선 카메라(회색 영상)와 RealSense 의 장치 노드는 건너뜁니다.
- 웹캠의 내부 파라미터는 보정값이 아니라 추정값입니다. 손목 이동이 일정 배율만큼 틀리고, 이 배율은 팔 속도로만 나타납니다(`launch/bimanual_sim_v2.launch.py` 와 같음).
- RealSense 를 옮겼다면 `tools/measure_camera_tilt.py` 를 다시 돌립니다.
- 손을 카메라에 20 cm 보다 가깝게 들면 깊이 센서가 손을 놓칩니다(`docs/results/r6_ego_depth_follow.md`).

## 실물 팔 (`--real`, `--dry`)

- 두 팔이 모두 `READY` 여야 시작합니다(`tools/check_pose_v2.py`). 아니면 아무것도 움직이지 않고 끝납니다.
- 설정은 R5/R6 에서 통과한 값 그대로입니다(`config/demo.yaml`).
  - 팔 끝은 시작 위치에서 7 cm 안, 손 움직임의 0.5 배
  - 관절 속도는 0.3 rad/s 이하, 관절은 시작 자세에서 0.3 rad 안
- 브리지는 `duration` 초 뒤 스스로 팔을 멈춥니다. 브리지가 끝나면 시연 전체가 끝납니다.
- 화면 맨 위 띠에 `REAL ARMS MOVING` 이 빨간색으로 뜹니다. `--dry` 이면 `REAL ARMS: DRY RUN (not moving)`, 시뮬레이션이면 `SIMULATION` 입니다.
- 시작할 때 2~3 초 동안 두 손을 가만히 두세요. 그 위치가 팔의 기준이 됩니다.
- 비상정지 버튼을 손이 닿는 곳에 두세요.

시뮬레이션만 돌릴 때는 팔이 잘 보이도록 더 넓게 따라갑니다(반경 15 cm, 1.0 배, 1.0 rad/s). 시뮬레이션 팔의 시작 자세는 실물 팔의 R6 시작 자세입니다.

## 구성

```
demo/run_demo.sh            환경(ROS 2 Jazzy, 손 드라이버, xArm 메시), 카메라 고르기, 실물 확인
demo/detect_camera.py       카메라 고르기
demo/demo_tracker_node.py   비전 노드 v4 + /demo/image (손 골격, 추적 상태, 모드 띠)
launch/demo.launch.py       팔 추종 v6 두 개, 손 리타깃팅 두 개, 시뮬레이션 손, RViz, (실물) 브리지 v3 두 개
config/demo.yaml            시뮬레이션 시작 자세, 추종/브리지 설정
config/demo.rviz            양팔 + 양손 + /demo/image
```

손 리타깃팅과 시뮬레이션 손은 [allegro_vision_teleop](https://github.com/jacoblee1214/allegro_vision_teleop) 을 씁니다(`hand_repo:=` 또는 `ALLEGRO_TELEOP_DIR`).

## 확인한 것 (2026-10-27)

| 경우 | 결과 |
|---|---|
| RealSense (USB 3, 깊이) + 시뮬레이션 | `/demo/image` 29 Hz, 팔 `joint_states` 100 Hz, 손 100 Hz |
| 노트북 웹캠 + 시뮬레이션 | `/demo/image` 29.9 Hz (화면 앞에 사람이 없을 때 확인) |
| 녹화 클립 `ego_depth_424` | 두 손 골격과 `tracking (depth)` 가 그려짐 |
| `real:=dry` | 두 브리지가 실물 팔의 자세를 읽고 그 자세에서 추종 시작, 명령은 보내지 않음, 30 s 뒤 전체가 정상 종료 |
| `--real` 의 확인 단계 | 두 팔 READY 확인 뒤 'yes' 가 아니면 시작하지 않음 |

`--real` 로 실물 팔을 움직이는 실행은 아직 하지 않았습니다. 이때 쓰는 노드와 설정은 R6 와 같습니다.
