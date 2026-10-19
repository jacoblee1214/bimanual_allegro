# 2단계: 양손 손 텔레옵 (시뮬레이션)

작성 2026-10-19. 로봇 없이 녹화 영상으로 확인한 결과입니다. 실물 두 손 연결은 아직 하지 않았습니다.

## 1. 구성

카메라 한 대(또는 녹화 영상)에서 비전 노드가 두 손을 찾고, 손마다 손 저장소의 한 손 파이프라인을 네임스페이스만 바꿔 한 벌씩 띄웁니다.

```
vision/hand_tracker_node.py
  ├─ /left/allegro/vision/world_landmarks  → /left  retargeting_node_v3 → sim_bridge_node → mock 손 (ros2_control)
  └─ /right/allegro/vision/world_landmarks → /right retargeting_node_v3 → sim_bridge_node → mock 손 (ros2_control)
```

손 저장소의 노드와 launch 는 복사하지 않고 그대로 실행합니다. 이 저장소의 `launch/bimanual_sim.launch.py` 는 실행 순서와 인자, 두 손의 위치만 정합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/v6f_manuse_teleoperation/install/setup.bash
cd ~/bimanual_allegro
ros2 launch launch/bimanual_sim.launch.py source:=data/clips/v2_cross.avi loop:=true gpu:=true
```

카메라로 하려면 `source:=0` 을 주면 됩니다. RViz 에는 두 손과 카메라 영상이 함께 뜹니다. 화면 오른쪽이 로봇 왼손이라, 카메라 영상 속 사람 손 배치와 같습니다.

손 바닥(base) 방향은 고정입니다. 이 단계에서는 손가락만 따라가고, 손목 위치와 방향은 3단계에서 팔이 맡습니다.

## 2. 손 저장소에서 바꾼 것

인수인계 때 허용된 범위("한 손 파이프라인을 네임스페이스로 여러 벌 띄울 수 있게")만 바꿨습니다. 네임스페이스를 주지 않으면 토픽 이름과 동작이 예전과 같습니다.

| 파일 | 변경 |
|---|---|
| `launch/allegro_hand_bringup.launch.py` | `namespace`, `frame_prefix`, `world_tf` 인자 추가. 스포너가 `controller_manager` 를 상대 이름으로 찾음. 드라이버의 컨트롤러 yaml 을 읽어 키를 `/**/<노드>` 로 넓힌 사본을 넘김 |
| `retargeting_node_v3.py` | 토픽을 상대 이름으로, `--ros-args` 를 받도록 |
| `sim_bridge_node.py` | 같음 |

컨트롤러 yaml 을 그대로 넘기면 네임스페이스 안의 `controller_manager` 가 설정을 못 찾습니다(`The 'type' param was not defined for 'joint_state_broadcaster'`). yaml 의 노드 이름이 루트 네임스페이스로 읽히기 때문입니다. 드라이버 패키지는 고치지 않는다는 규칙이 있어, launch 가 읽어서 키만 넓힌 사본을 임시 파일로 만듭니다. 값은 드라이버 것을 그대로 씁니다.

HANDOVER 6장은 관절 이름 충돌을 xacro 의 `ah_prefix` 로 풀자고 했습니다. 여기서는 그 방법 대신 네임스페이스와 TF 접두사를 썼습니다. `controller_manager` 가 손마다 따로 있으면 관절 이름이 같아도 토픽이 갈라져 충돌하지 않습니다. 겹치는 곳은 TF 하나뿐이라 `robot_state_publisher` 의 `frame_prefix` 로 충분합니다. 이렇게 하면 리타게터, 브리지, 컨트롤러 설정의 관절 이름을 하나도 바꾸지 않아도 됩니다.

## 3. 확인 결과 (2026-10-19, 녹화 영상 `v2_cross.avi`, RTX 5070 Ti, `--gpu`)

| 항목 | 결과 |
|---|---|
| 컨트롤러 | `/left`, `/right` 각각 `joint_state_broadcaster`, `allegro_hand_position_controller` active |
| 속도 | 비전 30 fps, 리타게터 손마다 약 30 Hz, 브리지 손마다 100 Hz |
| 좌우 | 리타게터와 같은 방식으로 chirality 를 계산. 왼손 토픽 467프레임 100% 왼손 모양(평균 -0.027), 오른손 토픽 412프레임 100% 오른손 모양(+0.029). `--no-undo-mirror` 가 맞음 |
| 지연 | 프레임을 읽은 시각부터 `target_joints` 도착까지 양손 중앙값 15.8 ms, p95 22~24 ms |
| 손이 가려질 때 | 브리지가 100 Hz 로 마지막 목표를 계속 보냄. 0.2~1.0 s 끊긴 구간 10개에서 150 ms 뒤 명령과 마지막 목표의 차이 최대 0.0015 rad. 손이 마지막 자세를 유지함 |
| 한 손 회귀 | 네임스페이스 없이 띄우면 토픽이 예전과 같은 이름(`/allegro/target_joints`, `/allegro_hand_position_controller/commands`)으로 나오고 브리지 100 Hz |
| RViz | 두 손이 `left/`, `right/` 프레임으로 나란히 그려짐 |

지연 15.8 ms 는 계산과 전송 시간입니다. 리타게터의 입력 필터(중앙값 5프레임 + EMA 0.3)와 브리지 EMA 가 만드는 신호 지연은 포함되지 않습니다.

## 4. 실물로 가기 전에 볼 것

- **손이 다시 나타날 때 튀는 움직임.** 0.3 s 넘게 끊기면 리타게터가 이전 해를 버리고 새로 풉니다. 이때 목표가 크게 바뀔 수 있고, 브리지 EMA(0.25, 100 Hz)는 이를 수십 ms 안에 따라갑니다. 시뮬레이션에서는 문제가 없지만 실물에서는 속도 제한이 필요할 수 있습니다.
- **실물 두 손 bring-up.** `hardware:=hardware` 와 손마다 다른 `descriptor` 로 같은 launch 를 두 번 띄우면 됩니다. 손은 Modbus 연결을 하나만 받으므로, 두 손 IP 가 겹치지 않는지 먼저 확인합니다. 왼손 브리지는 `--mode real --hand left` 로 띄우고, MCP 영점은 HANDOVER 4.3 을 따릅니다.
- **클러치와 대시보드.** `allegro/teleop_state` 도 손마다 나뉩니다. 대시보드는 아직 한 손용이고 카메라를 직접 엽니다. 양손용으로 바꿀 때는 비전 노드의 `/camera/image_raw` 를 구독하게 해야 카메라를 두 프로그램이 다투지 않습니다.
