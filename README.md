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

1단계 비전 레이어를 만드는 중입니다. 설계와 측정 기록은 [docs/01_vision_design.md](docs/01_vision_design.md), 선행 연구 조사는 [docs/survey_vision_hand_teleop.md](docs/survey_vision_hand_teleop.md) 에 있습니다.

```bash
# 환경: ROS 2 Jazzy, --system-site-packages 로 만든 venv 에 requirements.txt
source /opt/ros/jazzy/setup.bash
python -m pytest                                        # 하드웨어 없이 돌아갑니다

tools/calibrate_camera.py                               # 카메라 보정 (체커보드)
vision/hand_tracker_node.py --source 0 --camera-info config/camera_<name>.yaml
```
