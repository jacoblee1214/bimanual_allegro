# 에고 시점 카메라 (RealSense)

작성 2026-10-19. 노트북 정면 웹캠을 대신해, 휴머노이드 머리 카메라처럼 작업자 머리 뒤쪽 위에서 작업 공간을 내려다보는 카메라입니다.

## 1. 왜 바꾸나

- **좌표가 직관적입니다.** 정면 웹캠에서는 "작업자가 카메라를 마주 보고, 로봇은 작업자와 같은 쪽을 본다"는 가정을 머릿속에서 한 번 돌려야 했습니다. 에고 시점에서는 카메라가 곧 로봇 머리라서, 앞은 앞이고 왼쪽은 왼쪽입니다.
- **5단계와 시점이 같습니다.** 사람 작업 영상과 로봇 머리 카메라의 시점이 같아지면, 사람 영상에서 배우는 단계의 시점 차이가 줄어듭니다. 에고 시점 사람 영상을 로봇 학습에 쓰는 연구 흐름이 있습니다(예: EgoMimic, arXiv:2410.24221).
- **깊이를 직접 잽니다.** 단, USB 3 연결이 필요합니다(4장).

## 2. 설치

- 장치: Intel RealSense D457. SDK 에는 "RealSense D455" 로 보고되고, 펌웨어는 5.13.1.53, IMU 가 있습니다.
- 위치: 작업자 머리 뒤쪽 위에 고정합니다. 머리에 쓰면 고개를 돌릴 때 팔이 움직이므로 쓰지 않습니다.
- 환경: `.venv-vision` (requirements.txt + `pyrealsense2==2.59.1.11053`)
- 권한: librealsense v2.56.5 의 `config/99-realsense-libusb.rules` 를 `/etc/udev/rules.d/` 에 설치했습니다(sha256 `21dd6bc4…d7532a`, 사용자가 sudo 로 설치). 이것이 없으면 IMU 가 `Permission denied` 로 열리지 않습니다.

## 3. 기울기와 매핑

`tools/measure_camera_tilt.py` 는 정지한 카메라의 가속도를 2초간 평균합니다. 그 값을 장치의 IMU → 컬러 외부 파라미터로 컬러 카메라 좌표계로 옮겨 중력 방향을 구하고, 로봇 축을 정합니다.
- x: 카메라 시선을 수평으로 눕힌 방향(앞)
- z: 중력 반대(위)
- y: 왼쪽

결과는 `config/ego_camera.yaml`(`R_CB`)과 `config/ego_camera_info.yaml`(공장 내부 파라미터)에 저장됩니다. 카메라를 옮기면 다시 실행합니다.

- 측정(2026-10-19 17:11): 아래로 44.3°, 좌우 −4.9°, 샘플 간 편차 0.18°
- 수평 카메라와 45° 카메라를 넣어 축 계산을 검증했습니다.
- 컬러 640x480 공장 값: fx 383.7, fy 383.3, cx 321.4, cy 245.0
- 컬러 스트림의 왜곡 모델은 inverse Brown-Conrady 라서 OpenCV 모델과 방향이 반대입니다. 계수가 작아(화면 끝 약 0.6%) 0 으로 둡니다.

`arm/arm_follow_node_v5.py --camera config/ego_camera.yaml` 이 이 `R_CB` 를 씁니다.

## 4. USB

노트북과 허브 모두 USB 3 를 지원하는데, 카메라는 두 번 모두 USB 2(480 Mbps)로 붙었습니다. 케이블이 원인으로 보입니다.

| USB 2 에서 | 결과 |
|---|---|
| 컬러 640x480@30 단독 | 26 fps, 사용 가능 |
| 깊이 480x270@30 단독 | 27.7 fps |
| 컬러 + 깊이 15 fps | 11.2 fps |
| 컬러 640x480@30 + 깊이 848x480@30 | 시작 불가 |

그래서 지금은 **컬러 + IMU** 만 쓰고, 손목 위치는 정면 웹캠 때처럼 PnP 로 구합니다. USB 3 케이블을 구하면 깊이 보정을 붙입니다.

## 5. 좌우 판별

`docs/results/m3_ego_labels.md` 결과에 따라, 에고 시점에서는 라벨이 아니라 **위치로** 좌우를 정합니다(`vision/handedness_ego.py`, `vision/hand_tracker_ego.py`, 노드의 `--view ego`). 두 손을 교차하지 않는 것이 운용 조건입니다.

## 6. 실행

```bash
.venv-vision/bin/python tools/measure_camera_tilt.py                 # 카메라를 옮길 때마다
.venv-vision/bin/python vision/hand_tracker_node_v2.py --source realsense --view ego --gpu
.venv-arm/bin/python arm/arm_follow_node_v5.py --q0-from-robot --position-only --scale 0.5 \
    --radius 0.07 --joint-box 0.3 --max-joint-speed 0.3 --camera config/ego_camera.yaml --ros-args -r __ns:=/left
.venv-arm/bin/python arm/xarm_bridge_node_v3.py --side left --duration 30 [--enable] ... --ros-args -r __ns:=/left
```
