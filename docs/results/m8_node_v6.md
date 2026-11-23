# M8 결과: fusion v4 를 노드 v6 와 실제 로봇 스크립트에 연결

- 날짜: 2026-11-23
- 작성 에이전트와 모델: Kiro 서브에이전트(구현 담당), 모델 Auto(서버 자동 선택, 정확한 모델명은 확인하지 못함)
- 입력 데이터: `data/clips/ego_depth_424/`(녹화 영상, 실물 카메라는 쓰지 않음), `config/ego_camera_info.yaml`
- 원자료: `data/results/m8_node_v6_ros.json`(+ 노드 로그 `data/results/m8_node_v6_ros.node.log`, git 미포함). 아래 숫자는 모두 이 JSON 의 값입니다.

## 재현 명령

```bash
cd ~/bimanual_allegro
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
.venv-vision/bin/python -B tools/eval_node_v6_ros.py ego_depth_424      # 영상 길이(약 33 s) + 오프라인 재계산
.venv-vision/bin/python -B -m pytest -q tests
.venv-arm/bin/python -B -m pytest -q tests --ignore=tests/test_camera.py --ignore=tests/test_hand_tracker.py --ignore=tests/test_wrist_pose.py
```

## 1. 만든 파일

| 파일 | 역할 |
|---|---|
| `vision/hand_tracker_node_v6.py` | 노드 v5 + `--fusion v4`. 기본값은 v5 와 같은 `v1`(v4 로 바꾸지 않음). 노드 기반 클래스, 시간 기록, v1~v3 선택은 v4/v5 에서 import. `main()` 만 v5 를 복사(파서와 융합 선택이 안에 있어 재사용 불가) |
| `vision/wrist_diag.py` | `/<side>/wrist_depth_diag` 메시지 본문. v4 노드의 키를 그대로 두고, 융합 출력에 `gap_factor` 가 있을 때만 그 키를 더함(v1~v3 메시지는 키가 전과 같음). ROS 없이 import 되어 두 환경에서 시험 가능 |
| `tools/eval_node_v6_ros.py` | 노드 v6 `--fusion v4` 를 ROS 로 끝까지 돌려 오프라인 추적기(`WristDepthV4`)와 프레임별 비교 |
| `tools/run_real_ego_v2.sh` | `run_real_ego.sh` + `--fusion v2\|v3\|v4`(기본 v3). v4 일 때 노드 v6. **실행하지 않음** |
| `tests/test_hand_tracker_node_v6.py` | 시험 6개(아래) |

## 2. 오프라인 대비 일치 (`same_as_offline`)

노드는 영상을 기록된 속도로 한 번 재생하고 끝났습니다(`node_exit` = 0, ROS 도메인 77). 오프라인은 같은 영상을 ROS 없이 `DepthEgoHandTracker` + `WristDepthV4`(GPU MediaPipe, 같은 내부 파라미터)로 돌린 것입니다.

| 손 | 노드 발행(wrist_pose) | 오프라인 | diag 발행 | 최대 위치 차이 (mm) | source 일치 | pixels 일치 | gap_factor 일치 |
|---|---|---|---|---|---|---|---|
| 왼손 | 930 | 930 | 930 | 0.0 | 예 | 예 | 예 |
| 오른손 | 914 | 914 | 914 | 0.0 | 예 | 예 | 예 |

출처별 프레임 수(`diag.by_source`): 왼손 depth 908 / hold 4 / pnp 18, 오른손 depth 897 / hold 1 / pnp 16. `hold` 프레임의 `gap_factor` 범위는 왼손 1.0 ~ 1.0305, 오른손 1.0 ~ 1.0. 영상 입력 987 프레임(`/camera/image_raw`).

노드 시간(`node_timing_summary`): 29.9 fps, 추적 p50 7.65 ms / p95 9.1 ms(중앙값 기준).

## 3. 시험

- `.venv-vision`: 133 passed, 10 skipped(전: 129 / 8). `.venv-arm`: 80 passed, 26 skipped(전: 76 / 24).
- 새 시험 6개 중 4개는 ROS 없이 돕니다(diag 키 3개, 스크립트 분기 문자열 1개). 2개(노드의 융합 선택, `--help` 에 v4 와 기본값)는 rclpy, OpenCV, MediaPipe 가 없으면 `skipif` 로 건너뜁니다. 건너뛴 2개는 ROS 를 source 한 `.venv-vision` 에서 따로 돌려 6개 모두 통과했습니다.

## 4. `run_real_ego_v2.sh` 와 원본의 차이

머리말 주석(이름, `--fusion v2|v3|v4` 설명)과 다음 둘 뿐입니다. 안전 설정(반경 0.07, 0.5배, 0.3 rad/s, 관절 0.3 rad, `--duration` 뒤 정지, READY 확인, 'yes' 확인, 로그 폴더 중복 거부)은 한 줄도 바꾸지 않았습니다.

1. `--fusion` 값을 검사하고 `v2`·`v3` 는 `hand_tracker_node_v5.py`, `v4` 는 `hand_tracker_node_v6.py` 를 고름. 그 밖의 값은 로봇 그래프를 건드리기 전에 종료 코드 2 로 거절합니다(원본은 v1 등도 노드에 넘겼음).
2. 노드 실행 줄이 `"$NODE"` 를 씀.

`bash -n` 통과. `diff` 전문은 작업 보고에 있습니다.

## 5. 가정한 것과 하지 못한 것

- 이 영상에는 공백(`hold`)이 거의 없어(왼손 4, 오른손 1 프레임) v4 의 공백 처리는 노드 경로에서 거의 시험되지 못했습니다. 일치는 확인했지만, 긴 공백에서의 일치는 확인하지 못했습니다. 공백 길이에 따른 정확도는 `m8_gap_v4.md` 의 모의 실험이 근거입니다.
- 오프라인과 노드의 일치는 같은 코드 경로이기 때문에 0.0 mm 는 예상된 값입니다. 노드 연결(토픽, JSON 직렬화)과 `gap_factor` 가 실제로 전달되는지를 확인한 것이지 v4 의 품질을 확인한 것이 아닙니다.
- `run_real_ego_v2.sh` 는 실행하지 않았습니다(로봇 그래프). 문법 검사(`bash -n`)와 문자열 시험만 했습니다. 실물 카메라, 로봇, 실제 `--fusion v4` 구동은 하지 않았습니다.
- v4 의 gain 0.5 는 이 영상에 맞춰 고른 값이라는 `m8_gap_v4.md` 의 단서가 그대로 적용됩니다. 기본 융합을 v4 로 바꾸는 판단은 하지 않았습니다.
- 패키지 설치 없음, git 명령 없음, 기존 파일 수정 없음.
