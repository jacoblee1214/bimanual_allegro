# 에이전트 작업 규칙

이 저장소에서 일하는 에이전트(Claude, Gemini 등)는 모두 이 규칙을 따릅니다.

## 저장 위치

| 무엇 | 어디 | git |
|---|---|---|
| 라이브러리 코드 | `vision/`, `arm/` | 올림 |
| 실행 도구 | `tools/` | 올림 |
| 테스트 | `tests/` | 올림 |
| 설계 문서 | `docs/NN_<주제>.md` | 올림 |
| **결과 보고서** | `docs/results/<마일스톤>_<주제>.md` | 올림 |
| 측정 원자료(JSON, npz) | `data/results/<마일스톤>_<주제>.json` | 안 올림 |
| 녹화, 궤적, 모델 파일 | `data/clips/`, `data/trajectories/`, `data/models/` | 안 올림 |

결과 보고서에는 다음을 꼭 적습니다.
- 날짜
- 작성한 에이전트와 모델
- 입력 데이터
- 재현 명령
- 측정 표
- 가정한 것과 하지 못한 것

숫자는 원자료 JSON 에 있는 값만 씁니다.

## 버전

- 이미 있는 스크립트나 문서를 고칠 때는 새 버전 파일(`foo_v2.py`, `03_..._v2.md`)을 만들고 옛 파일은 그대로 둡니다.
- 새로 만드는 파일은 버전 없이 시작합니다.
- README 는 한 파일로 두고, 최신 버전을 가리키게 고칩니다.

## 환경

- 비전(mediapipe, numpy 1.x): `~/allegro_vision_teleop/.venv/bin/python`
- 팔(pinocchio, xArm SDK, numpy 2.x): `.venv-arm/bin/python`
- 테스트는 두 환경에서 각각 돌립니다. pinocchio 가 필요한 테스트는 `pytestmark = pytest.mark.skipif(...)` 로 건너뜁니다. 모듈 수준의 `pytest.importorskip` 은 쓰지 않습니다. ROS 의 launch_testing pytest 플러그인이 이를 전체 실행 건너뛰기로 바꿔, 다른 환경의 테스트가 하나도 돌지 않습니다(`tests/test_arm_ik.py` 참고).
- 패키지를 설치하지 않습니다. 필요하면 보고서에 적습니다.

## 금지

- git 명령(커밋, 푸시는 총괄 에이전트만 합니다)
- 로봇과 네트워크 접근(xArm, Allegro Hand, 그 IP 주소). 실물 작업은 총괄 에이전트가 사용자 승인을 받아 합니다.
- 공개 저장소이므로 IP 주소, 시리얼 번호, 원익 로고, 내부 문서를 파일에 넣지 않습니다.
- 손 저장소(`~/allegro_vision_teleop`)와 원익 드라이버(`~/v6f_manuse_teleoperation`) 수정
