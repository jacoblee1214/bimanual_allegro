# M6 결과: 깊이 보정을 실시간 비전 노드에 넣기 (`hand_tracker_node_v4.py`)

- 날짜: 2026-10-20
- 작성 에이전트와 모델: Claude subagent (Claude Opus 5.5), M6 노드 담당
- 입력 데이터
  - `data/clips/ego_depth_424/`: `tools/record_ego_depth.py` 로 녹화한 영상입니다. 컬러 640x480 과, 424x240 으로 받아 컬러에 정렬한 깊이로 이루어져 있으며 988 프레임입니다.
  - `data/results/m5_detections_ego_depth_424.npz`: M5 가 기록한 검출 결과입니다(재생 비교용).
  - `data/results/m5_depth_wrist.json`: M5 의 `fused_rk` 수치입니다(비교 기준).
- 원자료
  - `data/results/m6_node_v4.json`: ROS 없이 돌린 오프라인 비교와 처리 시간입니다.
  - `data/results/m6_node_v4_ros.json`: ROS 노드 전체 경로의 토픽 주기, 오프라인과의 일치 여부, 노드가 남긴 처리 시간 로그입니다. 노드 로그 원문은 `data/results/m6_node_v4_ros.node.log` 에 있습니다.
  - 이 문서의 숫자는 모두 위 JSON 에서 가져왔습니다.

## 재현 명령

```bash
cd ~/bimanual_allegro
# 1) ROS 없이: 추적기를 클립에 돌려 M5 fused_rk 와 비교하고, 처리 시간을 잽니다
.venv-vision/bin/python -B tools/eval_node_v4.py ego_depth_424
# 2) ROS 노드 전체 경로: 노드를 자식 프로세스로 띄워 모든 토픽을 받고, 오프라인 결과와 비교합니다
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
.venv-vision/bin/python -B tools/eval_node_v4_ros.py ego_depth_424
# 3) 테스트
.venv-vision/bin/python -B -m pytest -q tests
```

노드를 직접 띄우는 명령은 아래와 같습니다.

```bash
source /opt/ros/jazzy/setup.bash
# 실물 카메라 (총괄 에이전트가 실행합니다. 이 작업에서는 카메라를 열지 않았습니다)
.venv-vision/bin/python -B vision/hand_tracker_node_v4.py --source realsense --view ego --depth --fusion v1 --gpu
# 녹화 클립 재생 (카메라 없이 전체 ROS 경로를 시험할 때)
.venv-vision/bin/python -B vision/hand_tracker_node_v4.py --source clip:data/clips/ego_depth_424 --view ego --depth --fusion v1 --gpu [--loop]
```

`--depth` 를 빼면 v3 와 같이 동작합니다(`--ray-filter` 포함). `--fusion v2` 를 주면 `vision/depth_wrist_v2.py` 의 `WristDepthV2` 를 그 시점에 import 합니다.

---

## 요약

- 녹화된 검출을 그대로 넣어 재생하면, 새 추적기가 내는 손목 위치가 M5 의 오프라인 `fused_rk` 와 거의 같습니다. 프레임별 차이의 중앙값은 왼손 0.0084 mm, 오른손 0.0105 mm 입니다. `still` 단계 광선 방향 프레임 간 p95 는 왼손 21.04 mm(M5 21.03), 오른손 3.39 mm(M5 3.39)입니다.
- GPU MediaPipe 를 클립에 다시 돌려도 `still` 단계 수치가 같게 나왔습니다(왼 21.04, 오른 3.39 mm).
- ROS 노드를 클립으로 재생하면 `/left/wrist_pose` 930 개, `/right/wrist_pose` 914 개가 발행되었습니다. 그 위치는 ROS 없이 같은 클립에 돌린 추적기 결과와 **0.0 mm** 차이로 같았고, 진단 토픽의 `source` 순서도 모두 같았습니다. `/camera/image_raw` 는 988 장 모두 29.93 Hz 로 받았습니다.
- 처리 시간(검출 + PnP + 깊이 보정 + 필터)은 오프라인에서 p50 6.96 ms, p95 9.81 ms 였습니다. 노드에서 실시간 속도로 재생했을 때는 5 초 구간별 p50 의 중앙값이 8.8 ms, p95 의 중앙값이 11.05 ms 였습니다. 깊이 보정 때문에 늘어난 몫은 p50 기준 약 1.6 ms 입니다(2.16 − 0.58 ms).
- 실물 RealSense 경로(`--source realsense --depth`)는 카메라를 열지 않았으므로 **시험하지 못했습니다**.

---

## 1. 만든 파일

| 파일 | 역할 |
|---|---|
| `vision/hand_tracker_ego_v3.py` | `DepthEgoHandTracker`: 위치로 좌우를 나누고(`EgoSides`) PnP 를 푼 뒤, 깊이 보정 → 출처별 `sigma_ray` 를 쓰는 RayKalman → 회전은 v2 처럼 One Euro 를 거칩니다. 보정 방식은 `fusion_factory` 로 바꿀 수 있고, 기본값은 M5 를 감싼 `DepthFusionV1` 입니다 |
| `vision/depth_filter_v2.py` | `RayKalmanV2`: 측정마다 `sigma_ray` 를 인자로 받습니다. 필터 자신의 값은 바꾸지 않습니다 |
| `vision/rgbd_source.py` | `RealSenseRGBD`(컬러 640x480@30 + 깊이 424x240@30, `rs.align` 으로 컬러에 정렬하고 depth_scale 을 곱함)와 `ClipRGBD`(녹화 클립 읽기). ROS 를 쓰지 않으므로 테스트와 평가 도구가 함께 씁니다 |
| `vision/hand_tracker_node_v4.py` | v3 노드의 동작에 `--depth`, `--fusion v1\|v2`, `--source clip:<dir>` 를 더했습니다. 토픽은 v3 와 같고, `/<side>/wrist_depth_diag` 를 추가로 발행합니다 |
| `tests/test_hand_tracker_ego_v3.py` | 합성 손과 가짜 검출기로 만든 테스트 11 개입니다. 카메라와 MediaPipe 모델이 없어도 돕니다 |
| `tools/eval_node_v4.py` | ROS 없이 돌리는 비교와 처리 시간 측정 → `m6_node_v4.json` |
| `tools/eval_node_v4_ros.py` | ROS 노드 전체 경로 검증 → `m6_node_v4_ros.json` |

`rgbd_source.py` 와 두 평가 도구는 요청 목록에 없던 파일입니다. 클립 읽기를 노드에 두면 rclpy 가 없는 테스트 환경에서 시험할 수 없고, 평가 도구도 같은 코드를 다시 써야 해서 따로 떼었습니다.

### 1.1 보정 방식 인터페이스

`fusion_factory()` 는 손마다 한 번 불리고, 반환된 객체는 다음 두 가지를 갖춰야 합니다.

```python
__call__(pose: WristPose, image_landmarks, world_landmarks, depth_m: Optional[np.ndarray],
         camera_matrix, side) -> FusedWrist
reset()
```

- 반환값 `FusedWrist` 에 `rotation` 속성이 있고 None 이 아니면, PnP 회전 대신 그 회전을 One Euro 필터에 넣습니다.
- `pixels` 속성이 있으면 진단 토픽에 그 값을 싣습니다. `DepthFusionV1` 은 `palm_depth` 가 남긴 화소 수를 넣고, 다른 에이전트의 `WristDepthV2` 에는 이 속성이 없어 `null` 로 나갑니다.
- `source` 가 `"depth"` 인 프레임은 `sigma_ray` 5 mm(M5 의 `SIGMA_RAY_FUSED`), 그 밖의 프레임(`hold`, `pnp`)은 M4 의 45 mm 를 씁니다.
- 한 손이 `RESET_GAP_S`(0.3 s)보다 오래 보이지 않았다가 다시 잡히면, 그 손의 보정 객체에 `reset()` 을 부릅니다. 같은 손의 RayKalman 과 One Euro 필터가 초기화되는 조건과 같습니다.
- 깊이가 없으면(`depth_m=None`) 모든 프레임이 `hold` 또는 `pnp` 가 되어 45 mm 를 씁니다. `DepthFusionV1` 에서는 이것이 v2 추적기(`RayEgoHandTracker`)와 같은 동작이며, 테스트 `test_without_depth_it_is_the_v2_ray_filter` 가 이를 확인합니다.

다른 에이전트의 `vision/depth_wrist_v2.WristDepthV2` 는 작업 중에 생겼습니다. 이 인터페이스로 그대로 연결되는 것을 확인했습니다(오프라인에서 한 번 돌리고, 노드를 `--fusion v2` 로 몇 초 띄움). 그 수치는 이 문서의 대상이 아니므로 JSON 에 넣지 않았습니다. v2 의 결과는 `docs/results/m6_wrist_flip.md` 에 있습니다.

### 1.2 진단 토픽

`/<side>/wrist_depth_diag` (`std_msgs/String`, JSON)는 손목 자세를 발행한 프레임마다 나갑니다. 예시는 다음과 같습니다.

```json
{"source": "depth", "scale": 1.4263, "measured": 1.4263, "pixels": 846, "sigma_ray_mm": 5.0}
```

### 1.3 노드의 시간 기록

노드는 5 초마다 fps 와 함께 프레임당 시간의 p50/p95 를 로그로 남깁니다.

- `read`: 카메라 대기 + 정렬 + 깊이의 미터 변환입니다. 클립에서는 파일 읽기 시간입니다.
- `track`: 검출 + PnP + 깊이 보정 + 필터입니다.
- `publish`: 메시지 발행입니다.

클립을 재생할 때는 `frames.jsonl` 에 기록된 시각에 맞춰 프레임을 처리합니다. `--loop` 로 반복할 때는 시각이 계속 늘어나도록 한 바퀴의 길이만큼 더합니다.

---

## 2. 오프라인 일치 확인 (ROS 없음, `m6_node_v4.json`)

### 2.1 녹화된 검출을 재생한 경우 (`replay`)

M5 의 검출(`m5_detections_ego_depth_424.npz`)을 가짜 검출기로 넣고 클립의 깊이를 함께 주었습니다. 기준값은 M5 의 `fused_rk` 를 같은 검출에서 `tools/eval_depth_wrist.py` 와 같은 방법으로 다시 계산한 것입니다.

| 손 | 검출 프레임 | 검출 프레임 일치 | 위치 차이 p50 / p99 / 최대 (mm) | 센서 사용 여부가 다른 프레임 | 센서 보정 비율 |
|---|---|---|---|---|---|
| 왼 | 930 | 예 | 0.0084 / 0.0715 / 0.1139 | 0 | 0.9226 |
| 오른 | 914 | 예 | 0.0105 / 0.0479 / 3.0713 | 1 (t = 28.4335 s) | 0.9453 |

차이가 생긴 이유는 두 가지입니다.

- **내부 파라미터.** M5 는 PnP 에 `config/ego_camera_info.yaml`, 깊이 보정에 `meta.json` 의 컬러 내부 파라미터를 썼습니다. 추적기는 두 곳에 하나의 내부 파라미터(재생에서는 yaml)를 씁니다. 두 fx 의 차이는 0.1 px 안팎이며, p50 이 0.01 mm 수준인 것은 이 때문으로 봅니다 **[의견]**.
- **보정 객체 초기화.** 오른손은 `free` 단계에서 한 번 길게 사라졌습니다(ROS 경로에서 받은 오른손 토픽 간격 최대 503.9 ms). 추적기는 이때 보정 객체를 초기화하지만, M5 평가는 초기화하지 않았습니다. 그래서 그 뒤 최근 배율 창의 내용이 달라지고, 한 프레임(t = 28.4335 s)에서 상대 범위 판정이 갈렸습니다. 오른손 최대 차이 3.0713 mm 는 이 구간에서 나왔습니다. `still` 단계와는 겹치지 않습니다.

`still` 단계 떨림 (240 프레임, mm):

| 손 | 계열 | 광선 std | 옆 std | 광선 프레임 간 p95 | 옆 프레임 간 p95 |
|---|---|---|---|---|---|
| 왼 | 추적기 (`replay`) | 14.0 | 11.92 | 21.04 | 13.55 |
| 왼 | M5 다시 계산 | 13.99 | 11.92 | 21.03 | 13.56 |
| 왼 | M5 저장값 (`m5_depth_wrist.json`) | 13.99 | 11.92 | 21.03 | 13.56 |
| 오른 | 추적기 (`replay`) | 12.99 | 8.5 | 3.39 | 2.94 |
| 오른 | M5 다시 계산 | 12.99 | 8.5 | 3.39 | 2.94 |
| 오른 | M5 저장값 | 12.99 | 8.5 | 3.39 | 2.94 |

지연 (ms): 추적기의 옆 방향 / 광선 방향 지연은 왼손 4.0 / 9.0, 오른손 3.0 / 20.0 입니다. M5 저장값은 왼 4.0 / 9.0, 오른 3.0 / 19.0 입니다.

### 2.2 GPU MediaPipe 를 다시 돌린 경우 (`live`)

노드가 클립을 재생할 때처럼, 내부 파라미터는 `meta.json` 에서 읽었습니다. 비교 대상 `raw_rk` 는 같은 검출을 깊이 없이 같은 추적기에 다시 넣은 결과이며, v3 노드 `--ray-filter` 와 같은 동작입니다. 두 실행의 검출 프레임은 두 손 모두 같았습니다.

| 손 | 검출 | `depth` / `hold` / `pnp` | 센서 보정 비율 |
|---|---|---|---|
| 왼 | 930 | 858 / 54 / 18 | 0.9226 |
| 오른 | 914 | 864 / 34 / 16 | 0.9453 |

`still` 단계 떨림 (240 프레임, mm)과 지연 (ms):

| 손 | 계열 | 광선 std | 옆 std | 광선 프레임 간 p95 | 옆 프레임 간 p95 | 옆 지연 | 광선 지연 |
|---|---|---|---|---|---|---|---|
| 왼 | `raw_rk` (v3 노드) | 39.48 | 12.78 | 20.12 | 13.73 | 4.0 | 59.0 |
| 왼 | `fused_rk` (v4 노드) | 14.0 | 11.92 | 21.04 | 13.56 | 4.0 | 9.0 |
| 오른 | `raw_rk` (v3 노드) | 25.04 | 7.0 | 7.45 | 2.34 | 3.0 | 52.0 |
| 오른 | `fused_rk` (v4 노드) | 13.0 | 8.5 | 3.39 | 2.94 | 3.0 | 20.0 |

이 수치는 2.1 의 재생 결과와 0.01 mm 이내로 같습니다. GPU 검출이 녹화 당시와 같은 결과를 냈고, 남은 차이는 내부 파라미터(재생은 yaml, 여기서는 `meta.json`) 차이입니다 **[의견]**. 왼손의 광선 프레임 간 p95 가 `raw_rk` 보다 나아지지 않은 것(20.12 → 21.04 mm)은 M5 보고서 5.2 의 PnP 자세 전환 때문이며, v1 보정 방식으로는 고쳐지지 않습니다.

### 2.3 처리 시간 (오프라인, 속도 제한 없이 돌림, 처음 10 프레임 제외)

| 항목 | p50 (ms) | p95 (ms) | 최대 (ms) | n |
|---|---|---|---|---|
| `process()` 전체 (검출 + PnP + 보정 + 필터) | 6.96 | 9.81 | 62.87 | 978 |
| 그중 검출(MediaPipe GPU) | 4.77 | 7.63 | 60.92 | 978 |
| 검출을 뺀 나머지, 깊이 있음 | 2.16 | 2.69 | 3.2 | 978 |
| 같은 추적기, 녹화 검출, 깊이 없음 (색 변환 + PnP + 필터) | 0.58 | 0.67 | 0.85 | 978 |
| 깊이 uint16 → 미터 float32 변환 (640x480) | 0.103 | 0.121 | | 247 |

- 깊이 보정은 두 손 합쳐 프레임당 약 1.6 ms 를 더합니다(2.16 − 0.58, p50). 대부분은 `palm_depth` 가 손바닥 다각형 안쪽 화소를 고르는 데 쓰입니다 **[의견, 따로 재지 않음]**.
- 검출의 최대값(60.92 ms)은 드물게 나오는 지연입니다. p95 는 7.63 ms 입니다.
- RealSense 경로에서 `rs.align` 에 드는 시간은 카메라가 없어 재지 못했습니다(5장).

---

## 3. ROS 노드 전체 경로 (`m6_node_v4_ros.json`)

`tools/eval_node_v4_ros.py` 가 노드를 자식 프로세스로 띄웠습니다.

```
vision/hand_tracker_node_v4.py --source clip:<repo>/data/clips/ego_depth_424 --view ego --depth --fusion v1 --gpu
```

`--loop` 없이 클립을 한 번 재생하고 노드가 스스로 끝나도록 했습니다(종료 코드 0). ROS 도메인은 77, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, RMW 는 기본값입니다. 저장소의 다른 노드도 RMW 를 따로 지정하지 않습니다.

### 3.1 토픽 주기 (수신 시각 기준)

메시지를 역직렬화하지 않고 도착 시각만 셌습니다. `ros2 topic hz` 도 따로 돌려 보았는데, 포즈와 진단 토픽에서는 30 Hz 근처를 보였습니다. 하지만 900 kB 이미지 토픽에서는 `ros2 topic hz` 자신이 메시지를 따라가지 못해 훨씬 낮은 값을 보였습니다. 그래서 표에는 도착 시각으로 센 값을 씁니다(`ros2 topic hz` 결과는 JSON 에 넣지 않았습니다).

| 토픽 | 메시지 | 주기 (Hz) | 간격 p50 / p95 / 최대 (ms) |
|---|---|---|---|
| `/camera/image_raw` | 988 | 29.93 | 33.4 / 45.8 / 74.6 |
| `/left/wrist_pose` | 930 | 29.92 | 33.4 / 36.7 / 106.5 |
| `/left/wrist_depth_diag` | 930 | 29.92 | 33.4 / 36.7 / 106.5 |
| `/left/allegro/vision/world_landmarks` | 930 | 29.92 | 33.4 / 36.8 / 106.6 |
| `/right/wrist_pose` | 914 | 29.47 | 33.4 / 36.8 / 503.9 |
| `/right/wrist_depth_diag` | 914 | 29.47 | 33.4 / 36.7 / 503.9 |
| `/right/allegro/vision/world_landmarks` | 914 | 29.47 | 33.4 / 36.8 / 503.9 |

손 토픽의 최대 간격은 클립에서 그 손이 검출되지 않은 구간입니다. 이미지 토픽은 988 프레임을 빠짐없이 받았습니다.

### 3.2 오프라인과의 일치

| 손 | 발행된 손목 자세 | 오프라인 추적기 | 위치 최대 차이 (mm) | 진단 `source` 순서 일치 | 발행된 센서 보정 비율 |
|---|---|---|---|---|---|
| 왼 | 930 | 930 | 0.0 | 예 | 0.9226 |
| 오른 | 914 | 914 | 0.0 | 예 | 0.9453 |

토픽 이름, 메시지 형식, 좌표계(`camera_optical`)는 v3 와 같습니다. 따라서 `arm/arm_follow_node_v6.py` 는 고치지 않고 그대로 받을 수 있습니다. 다만 팔 노드와 함께 띄워 보지는 않았습니다.

### 3.3 노드가 기록한 처리 시간 (실시간 속도 재생, 5 초 구간 6 개)

| 항목 | 구간별 p50 의 중앙값 (최대) | 구간별 p95 의 중앙값 (최대) |
|---|---|---|
| fps | 29.9 (29.9) | |
| `read` (클립 읽기 + 미터 변환) | 1.15 (1.3) | 1.45 (1.7) |
| `track` (검출 + PnP + 보정 + 필터) | 8.8 (9.0) | 11.05 (12.8) |
| `publish` | 0.8 (0.8) | 0.95 (1.1) |

실시간 속도에서는 `track` 이 오프라인(p50 6.96 ms)보다 약 2 ms 느렸습니다. 프레임 사이에 GPU 가 쉬면서 클럭이 내려가기 때문으로 추정합니다 **[의견, 확인하지 않음]**. 그래도 33 ms 프레임 간격 안에 충분히 들어갑니다.

---

## 4. 테스트

| 명령 | 결과 |
|---|---|
| `.venv-vision/bin/python -B -m pytest -q tests` | **115 passed, 8 skipped** |
| 같은 명령, 이 작업의 테스트 파일만 빼고 | 104 passed, 8 skipped. 작업 전에는 95 passed, 8 skipped 였고, 그사이 다른 에이전트가 만든 `tests/test_depth_wrist_v2.py` 9 개가 더해졌습니다 |
| `tests/test_hand_tracker_ego_v3.py` 만 | 11 passed |

`tests/test_hand_tracker_ego_v3.py` 는 모듈 수준의 `pytestmark = pytest.mark.skipif(...)` 로 OpenCV 나 MediaPipe 가 없는 환경에서 건너뜁니다. 모듈 수준 `importorskip` 은 쓰지 않았습니다. `.venv-arm` 에서는 돌리지 않았습니다. 이 추적기는 OpenCV 와 MediaPipe 가 필요해 그 환경에서는 모두 건너뛰게 됩니다.

뮤테이션 확인(`python -B`, 원본으로 되돌린 뒤 `cmp` 로 확인):

| 바꾼 것 | 실패한 테스트 수 |
|---|---|
| 출처와 상관없이 항상 45 mm | 3 |
| 보정 방식이 준 회전을 무시 | 1 |
| 공백 뒤 `reset()` 을 부르지 않음 | 1 |
| RayKalman 에 측정별 sigma 를 넘기지 않음 | 2 |
| RayKalman 에 보정 전 PnP 위치를 넣음 | 3 |
| `RayKalmanV2` 가 자기 sigma 를 되돌리지 않음 | 1 |
| 진단의 `pixels` 를 비움 | 2 |
| 클립 깊이에 depth_scale 을 곱하지 않음 | 1 |

---

## 5. 가정한 것과 하지 못한 것

- **실물 카메라 경로는 시험하지 못했습니다.** 지시에 따라 RealSense 를 열지 않았습니다. `RealSenseRGBD` 는 `tools/record_ego_depth.py` 의 스트림 설정과 `rs.align` 사용법을 그대로 따랐지만 실행해 보지 않았습니다. 다음은 총괄 에이전트가 실물에서 확인해야 합니다.
  - `--source realsense --view ego --depth --gpu` 로 띄웠을 때 로그의 fps 가 30 근처인지 봅니다.
  - `read` 시간(대기 + `rs.align` + 미터 변환)이 얼마인지 봅니다. 총괄 에이전트의 간이 시험에서는 정렬까지 포함해 약 30 fps 가 나왔다고 전달받았습니다(총괄 에이전트의 값).
  - 깊이 진단 토픽의 `source` 가 대부분 `depth` 인지 봅니다.
- USB 3 가 아니면 `RealSenseRGBD` 는 녹화 도구와 같이 종료합니다.
- RealSense 의 내부 파라미터는 v3 와 같이 장치의 공장값에서 왜곡을 뺀 값입니다. 클립 재생에서는 `meta.json` 의 컬러 내부 파라미터를 씁니다. `--camera-info` 를 주면 그 값이 우선합니다.
- 깊이 보정 객체를 0.3 s 공백 뒤에 초기화하는 것은 이 작업에서 정한 것입니다. M5 평가는 초기화하지 않았고, 이 클립에서는 오른손 한 프레임에서만 결과가 갈렸습니다(2.1).
- `sigma_ray` 5 mm / 45 mm 와 M5 의 문턱 값은 그대로 두었고, 다시 튜닝하지 않았습니다.
- 왼손 PnP 자세 전환(M5 5.2)은 v1 보정 방식으로는 남아 있습니다. `--fusion v2` 가 이를 겨냥하지만, 그 평가는 다른 에이전트의 몫입니다.
- `arm/arm_follow_node_v6.py` 와 함께 띄우는 시험은 하지 않았습니다. 로봇과 네트워크에는 접근하지 않았습니다. ROS 는 도메인 77, localhost 범위에서만 썼고, 띄운 프로세스는 PID 로 직접 종료했습니다.
- 패키지는 설치하지 않았고, 기존 파일도 고치지 않았습니다.

## 총괄 검증 (Claude Opus 5.5, 2026-10-20)

- 실물 RealSense 경로를 총괄이 확인했습니다(`--source realsense --view ego --depth --fusion v1 --gpu`, ROS 도메인 77, 로봇 없음). 29.4 fps 였고 ms p50/p95 는 read 19.9/25.3(프레임 대기 포함), track 12.8/17.9, publish 0.5/0.6 이었습니다. 진단 토픽도 발행됐습니다.
- 그때 카메라는 에고 위치에 있지 않았습니다. 에고 시점에서 `depth` 비율이 얼마인지는 R6 전에 다시 봅니다.
- `tests/test_hand_tracker_ego_v3.py` 가 arm venv 에서 수집 오류(`Rotation` 미정의)를 냈습니다. 모듈 수준 상수를 `HAS_VISION` 일 때만 만들도록 고쳤습니다. 이제 arm 에서는 건너뜁니다.
