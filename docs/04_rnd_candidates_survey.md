# R&D 후보 조사: 우리 측정에서 나온 다섯 가지 문제와 선행 연구

2026-10-19 작성. 01~03 문서의 측정에서 드러난 다섯 가지 약점마다, 선행 연구가 무엇을 했는지와 어떤 효과를 보고했는지 조사했습니다. 그 결과를 바탕으로 후보의 순위를 매기고, 노트북 웹캠과 xArm7, Allegro 하드웨어에서 바로 잴 수 있는 실험을 제안합니다. 기존 조사 문서(`survey_vision_hand_teleop.md`)와 겹치는 내용은 필요한 만큼만 다시 적었습니다.

**표기.** 주장마다 근거의 성격을 붙였습니다.

| 표기 | 뜻 |
|---|---|
| **[확인]** | 논문 본문이나 초록 페이지를 직접 열어 확인했습니다 |
| **[검색]** | 검색 결과의 요약만 봤습니다. 인용하기 전에 원문을 확인해야 합니다 |
| **[실측]** | 이 저장소에서 측정한 값입니다(01, 02 문서) |
| **[의견]** | 조사와 측정을 바탕으로 한 판단입니다 |

"새로운 시도"인지에 대해서는 "검색 범위에서 찾지 못함"이라고만 쓰고, 그때 쓴 검색어를 부록 A 에 남겼습니다. 찾지 못했다는 것이 없다는 뜻은 아닙니다.

---

## 요약

- **가장 먼저 할 일은 연구가 아니라 안전 장치입니다.** 손이 0.3 s 넘게 사라졌다가 다시 나타날 때 목표가 튀는 문제는 속도 제한, 시간에 걸친 보간, 클러치 재고정으로 이미 풀려 있는 문제입니다. 새로움은 없지만 실물 전에 반드시 해야 합니다.
- **연구 후보 1순위는 "손 크기(배율)를 세션 동안 고정"하는 실험입니다.** 클러치 방식(03 문서)에서는 배율의 상수 오차는 이동 배율 10% 오차로만 나타나고, 문제가 되는 것은 프레임마다 흔들리는 배율입니다. 같은 아이디어를 HaWoR 출력에 적용한 사례는 있지만 MediaPipe + PnP 텔레옵에 적용한 사례는 검색 범위에서 찾지 못했습니다. 녹화해 둔 영상만으로 바로 잴 수 있고, 손등일 때 떨림이 4배인 문제(측정 3)의 원인도 함께 가려낼 수 있습니다.
- **2순위는 좌우 라벨을 "추적 정체성 + 3D 손 모양의 chirality"로 정하는 실험입니다.** 가장자리에서 라벨이 틀릴 때 MediaPipe 의 3D 손 모양까지 뒤집혀 있는지는 검색 범위에서 보고를 찾지 못했습니다. 이것도 녹화 영상으로 잴 수 있고, 결과에 따라 지금 "가운데 60%"로 묶인 작업 범위를 넓힐 수 있습니다.
- **모델 교체(WiLoR 등)는 정확도 이득이 크다는 보고가 있지만**, 그 수치는 대부분 손목 위치와 배율을 지운 지표(PA-MPJPE)입니다. 우리 문제인 손목 깊이와 떨림에 대한 이득은 녹화 영상으로 직접 재 봐야 압니다.
- **단안 깊이 네트워크는 낮은 순위입니다.** V100 에서도 한 장에 91~1300 ms 가 걸리고, 장면 단위 정확도도 손 크기 사전 지식보다 낫다고 보기 어렵습니다.

---

## 1. 우리 측정 다시 보기

| | 측정 | 출처 |
|---|---|---|
| 1 | PnP 로 구한 거리는 MediaPipe 가 추정한 손 크기에 비례합니다. 크기가 10% 틀리면 거리도 약 10% 틀립니다 | 01 문서 3.3 [실측, 합성] |
| 2 | 좌우 라벨이 x < 0.2, x > 0.8 에서 31% 틀렸고(177/570), 틀린 라벨의 점수가 최대 0.995 였습니다. 가운데 60% 안에서는 0/530 이었습니다 | 01 문서 V2 [실측] |
| 3 | 손등이 카메라를 향하면 손목 떨림이 19.5 mm 로, 손바닥일 때 4.6 mm 의 4배입니다 | 01 문서 V5 [실측] |
| 4 | MediaPipe Tasks API 는 RTX 5070 Ti 에서 두 손 3.7~4.2 ms 입니다 | 01 문서 V5 GPU [실측] |
| 5 | 0.3 s 넘게 끊기면 리타게터가 이전 해를 버리고 새로 풀어서 목표가 튈 수 있습니다. 손목 필터도 같은 0.3 s 기준으로 초기화됩니다 | 02 문서 4장, `vision/hand_tracker.py` 의 `RESET_GAP_S` [실측] |

한 가지를 먼저 짚습니다. 측정 2 의 31% 는 MediaPipe 가 낸 **원래 라벨**의 오류율입니다. 지금 규칙(같은 라벨이 두 번 나오면 x 위치로 정함, 01 문서 3.2)을 거친 **최종 배정**의 오류율은 따로 세지 않았습니다. 후보를 고르기 전에 이 값을 먼저 세야 합니다(6.2 의 첫 단계).

---

## 2. 후보별 선행 연구

### 2.1 손목 깊이와 배율 (측정 1)

**문제의 구조.** PnP 에 넣는 3D 점(MediaPipe world landmark)을 k 배 하면, 같은 영상 좌표를 설명하는 해는 회전은 그대로이고 평행이동만 k 배가 됩니다. 깊이뿐 아니라 좌우, 위아래 위치도 k 배가 됩니다 **[의견, 사영 기하에서 바로 나오는 성질]**. 그래서 클러치 방식에서는 두 종류의 오차를 나눠 봐야 합니다.

- **세션 동안 일정한 배율 오차**: 손 이동량 전체가 같은 비율로 커지거나 작아집니다. 작업자가 로봇을 보며 손을 움직이므로 이동 배율 `s` 가 10% 틀린 것과 같고, 크게 해롭지 않습니다 **[의견]**.
- **프레임마다 흔들리는 배율**: 손이 가만히 있어도 로봇이 앞뒤로 움직입니다. 손이 60 cm 거리에 있을 때 배율이 3% 흔들리면 깊이 방향으로 약 18 mm 의 가짜 움직임이 생깁니다 **[의견, 계산]**. 측정 3 의 떨림과 같은 크기입니다.

| 접근 | 선행 연구 | 보고된 효과 | 근거 |
|---|---|---|---|
| 손 크기 사전 지식으로 절대 깊이 복원 | Iqbal 외(ECCV 2018)는 손 자세를 배율을 뺀 형태로 추정하고, "손 크기에 대한 사전 지식이 주어지면" 배율을 따로 구할 수 있다고 씁니다 | 원리 제시 | [확인] |
| 약원근 배율에서 깊이 | AnyTeleop 은 RGB 한 대일 때 FrankMocap 처럼 손의 약원근 배율을 예측하는 네트워크를 더해 손목 위치를 구합니다. 같은 피아노 과제에서 오류율이 RGB 28.1%, RGB-D 21.8%, RGB-D 두 대 12.5% 였습니다(오류율은 틀린 건반 수 / 전체 건반 수) | RGB-D 로 오류율 약 1/4 감소 | [확인] |
| 같은 방식의 실패 사례 | Vuong 외(2021)는 깊이를 `C·f / s_h` 로 근사했고, RGB 만으로는 집어 옮기기 3회 중 1회만 성공했습니다. 실패 원인으로 잡음 섞인 깊이 추정과, 엄지와 검지 끝이 가까워질 때 깊이 추정이 무너지는 현상을 듭니다 | 손 모양이 바뀔 때 깊이가 흔들림 | [확인] |
| 2D/3D 정합 최적화 (우리와 같은 계열) | OAT(ICRA 2026)는 MediaPipe 의 2D 점과 손 기준 3D 점을 맞춰 손목 위치만 최적화하고, 회전은 손바닥 5점 평면으로 따로 구합니다. 이전 프레임과의 위치 차이에 벌점을 줍니다. FreiHAND 에서 MediaPipe 입력으로 잰 손목 위치 오차는 정확한 내부 파라미터일 때 11.41 cm, 보정 최적화 없이 7.647 cm, 보정 최적화를 하면 4.277 cm 였고, MediaPipe 가 검출한 이미지는 15.33% 였습니다 | MediaPipe 입력에서 손목 절대 위치 오차가 수 cm 수준 | [확인] |
| 손 모양이 바뀔 때의 흔들림 보정 | OAT 는 물건을 잡는 동안 손 모양 변화 때문에 손목 추정이 흔들리는 것을 관찰하고, 손 모양이 빠르게 바뀌는 구간에서 바이어스 항으로 손목을 붙잡는 "grasp stabilization"을 넣었습니다. 저역 통과 필터만으로는 부족했다고 씁니다 | 정성적 그래프 (수치 없음) | [확인] |
| 작업자별 손 모양 보정 | Kong 외(ECCV 2022)는 라벨 없는 RGB 몇 장으로 새 작업자의 MANO 형상 파라미터를 보정하고, 이를 네트워크에 입력으로 넣습니다. 형상을 아는 쪽이 복원이 정확하다고 보고합니다 | 초록에 수치 없음 | [확인] |
| 시퀀스 안에서 배율 고정 | Dexterous Point Policy(2026) 부록 F 는 HaWoR 출력에서 한 영상 안에서도 손 배율이 10 cm 넘게 변하고, 큰 손은 멀리, 작은 손은 가깝게 놓는 경향과 깊이 축 잡음이 크다는 점을 보고합니다. 형상 파라미터 β 를 출력이 아니라 입력으로 바꾸고, 추론 때는 HaMeR 의 프레임별 β 평균을 넣습니다 | 문제와 해법 모두 우리와 같은 구조 | [확인] |
| 자세별 손 크기 보정 (특허) | Snap 특허 공개(US 20260073545)는 대표 뼈 길이로 손 크기를 정하고, 이를 단안 추적에서 상대 깊이를 절대 깊이로 바꾸는 데 씁니다. 자세마다 여러 프레임에 걸쳐 측정을 반복해 변화가 1~3% 아래로 수렴하면 확정합니다 | 특허라 성능 수치 없음 | [확인, 특허 요약 페이지] |
| 팔뚝으로 배율 단서 | EgoForce(2026)는 팔뚝 모양을 거리 단서로 써서 HOT3D 에서 카메라 좌표 MPJPE 를 최대 28% 줄였다고 보고합니다 | 머리 착용 카메라 기준 | [확인] |
| 카메라 좌표 직접 예측 | HandDGP(ECCV 2024)는 카메라 좌표 손 메시를 한 번에 예측합니다. ScaleHP(2026)는 배율 토큰을 따로 두어 FreiHAND 카메라 좌표 MPJPE 35.8 mm 를 보고합니다. Ren 외(2026)는 HO3D 에서 카메라 좌표 오차를 최대 37.1% 줄였다고 보고합니다 | 학습 기반, 최고 수준도 3~4 cm | [확인] (HandDGP 는 초록만, 수치 미확인) |
| 단안 깊이 네트워크 | Depth Pro 논문의 표에서 V100 기준 한 장 처리 시간은 Depth Anything v2 90.9 ms, UniDepth 178.5 ms, Depth Pro 341.3 ms, Metric3D v2 1299.6 ms 입니다. 영점 학습 δ1(오차 25% 이내 화소 비율)은 Sun-RGBD 에서 72.4 ~ 95.8, 다른 데이터셋에서는 5.9 ~ 89.0 으로 편차가 큽니다 | 실시간 손목 깊이에는 느리고 부정확 | [확인] |
| 깊이 네트워크 + 시퀀스 최적화 | HaWoR(CVPR 2025)는 Metric3D 로 SLAM 의 배율을 맞추고, 실시간과는 거리가 멀다고 스스로 씁니다. Large Video Planner 는 HaMeR 의 평행이동이 시간에 따라 흔들리고 떠돌아서, MegaSaM 의 깊이로 다시 구합니다 | 오프라인 처리용 | HaWoR [확인], LVP [검색] |

**정리 [의견].**

- 배율 문제를 학습 모델로 푸는 연구는 많지만, 최신 카메라 좌표 오차도 3~4 cm 입니다(ScaleHP). 우리 PnP 의 상수 오차를 줄이는 데 큰 모델을 들일 이유는 약합니다. 클러치 방식에서는 상수 오차가 크게 해롭지 않기 때문입니다.
- 실제로 해로운 것은 배율의 시간 변화이고, 이에 대한 가장 직접적인 선행 사례는 "형상(β)을 시퀀스 동안 고정"(Dexterous Point Policy)과 "잡는 동안 손목 고정"(OAT)입니다. 둘 다 우리 파이프라인에 그대로 옮길 수 있습니다.
- MediaPipe world landmark 를 PnP 에 넣기 전에 세션 고정 배율로 다시 맞추는 방식은 검색 범위에서 찾지 못했습니다(부록 A, Q1~Q4, Q24). 새로운 방법이라기보다 알려진 아이디어를 우리 구성에 적용하는 것입니다.

### 2.2 좌우 라벨 (측정 2)

| 접근 | 선행 연구 | 보고된 효과 | 근거 |
|---|---|---|---|
| MediaPipe 의 구조 | 손 랜드마크 모델은 잘린 손 영상 한 장에서 21점, 손 존재 확률, 좌우 이진 분류를 함께 냅니다. 좌우 머리는 좌우 라벨을 단 실제 영상 일부로 학습했습니다. 영상 스트림에서는 이전 프레임의 랜드마크로 잘라낼 영역을 정하고, 손을 놓쳤을 때만 손바닥 검출기를 다시 돌립니다. 상대 깊이는 합성 영상으로만 학습했습니다 | 좌우는 프레임마다, 잘린 영역 하나만 보고 정합니다 | [확인] |
| MediaPipe 문서 | 좌우 라벨은 입력이 거울상이라고 가정하고 정합니다. world landmark 는 손의 대략적인 기하 중심을 원점으로 한 미터 단위 좌표입니다 | 우리 실측과 일치(Tasks API 는 반대) | [확인], [실측] |
| 추적으로 정체성 유지 | SORT(ICIP 2016)는 칼만 필터와 헝가리안 할당만으로 다른 온라인 추적기와 비슷한 정확도를 260 Hz 로 냈습니다 | 범용 물체 추적 | [확인] |
| 위치로 좌우 정하기 | 1인칭 영상 연구들은 SORT 로 손을 추적한 뒤 화면 위치로 좌우를 정합니다 | 1인칭 시점 전제 | [검색] |
| 손이 잠시 사라질 때 | TouchDesigner 사용자 포럼에 한 손이 잠깐 사라졌다 돌아오면 두 손의 번호가 뒤바뀐다는 보고가 있습니다 | 비공식 보고 | [검색] |
| 3D 점의 기하로 chirality 판정 (특허) | Snap 특허 공개(US 20260064192)는 검지와 중지 MCP 를 잇는 기준 벡터를 두고, 각 손가락 마디가 굽는 방향의 부호를 모아 왼손과 오른손을 가립니다. 편 손처럼 합이 0 에 가까우면 믿지 않고, 학습 분류기와 교차 확인할 수 있다고 씁니다 | 특허라 성능 수치 없음 | [확인, 특허 요약 페이지] |
| 메시 법선으로 판정 | HandOS(CVPR 2025)는 좌우 분류를 따로 두지 않는 한 단계 모델이며, 메시 법선으로 좌우를 정해 HInt 에서 97.9%(ViTPose 93.8%)를 냈다는 요약이 있습니다 | 원문에서 이 수치는 확인하지 못함 | 모델 [확인], 좌우 수치 [검색] |
| 영상 단위 모델 | 비디오 확산 모델로 손 동작을 복원한 2026년 연구는 좌우별 F1 을 따로 재며, 기존 방법이 없는 손을 만들어 내는 오류를 지적합니다 | 오프라인 | [검색] |

이 저장소의 손 저장소 리타게터는 이미 3D 점으로 chirality 를 계산해 반대 손이 들어오면 경고합니다. 02 문서에서 같은 계산으로 왼손 토픽 467프레임이 모두 왼손 모양(평균 -0.027), 오른손 토픽 412프레임이 모두 오른손 모양(+0.029)이었습니다 **[실측]**. 다만 이 측정은 가운데 영역의 영상에서 한 것이고, 가장자리의 틀린 라벨 프레임에서는 재지 않았습니다.

**정리 [의견].**

- MediaPipe 는 좌우를 잘린 영역 한 장만 보고 정하므로, 손이 가장자리에서 잘리면 단서가 줄어드는 구조입니다. 가장자리 오류와 높은 점수에 대한 공식 보고는 검색 범위에서 찾지 못했습니다(Q7, Q22, Q23).
- 핵심 질문은 **라벨이 틀린 프레임에서 3D 손 모양까지 뒤집혀 있는가**입니다. MediaPipe 의 3D 출력은 같은 특징에서 나오므로 같이 뒤집혀 있을 수도 있고, 좌우 머리만 틀렸을 수도 있습니다. 같이 뒤집혀 있다면 chirality 판정은 쓸모가 없고, 추적 정체성만 남습니다. 이 질문에 답한 보고는 검색 범위에서 찾지 못했습니다(Q8, Q25).
- 추적으로 정체성을 이어 가는 방법(SORT 계열)은 잘 알려져 있습니다. 우리 상황에서는 손이 두 개뿐이고, 교차할 때 MediaPipe 라벨이 오히려 위치보다 나았으므로(01 문서 V2), "트랙이 시작될 때 좌우를 정하고, 트랙이 이어지는 동안 라벨을 투표로만 반영"하는 규칙이 맞습니다.

### 2.3 손등일 때의 떨림 (측정 3)

| 접근 | 선행 연구 | 보고된 효과 | 근거 |
|---|---|---|---|
| 속도 적응 필터 | One Euro 필터(CHI 2012)는 속도에 따라 차단 주파수를 바꿉니다. 느릴 때는 낮춰 떨림을 줄이고, 빠를 때는 높여 지연을 줄입니다 | 지금 쓰는 필터 | [확인] |
| 신뢰도 가중 One Euro | Adzhiev, Hosken(SCA 2026 포스터)은 One Euro 와 신뢰도 가중 필터를 합친 방법을 단안 자세 추정에 썼습니다 | 초록에 수치 없음 | [확인] |
| 관측 잡음을 신뢰도로 조절 | 다중 물체 추적에서는 검출 신뢰도로 칼만 필터의 관측 잡음을 키우거나 줄이는 변형이 흔합니다. 사람 관절 추적에서 이를 쓴 HRI 필터(Martini 외 2024)가 있습니다 | 손목이 아닌 전신 | [검색] |
| 시점에 따른 오차 | OAT 는 Blender 로 그린 손을 여러 방향에서 보고 2D/3D 정합 오차를 지도로 그렸고, 위에서 내려다보거나 아래에서 올려다보는 시점이 가장 정확했다고 씁니다 | 시점에 따라 오차가 크게 달라짐 | [확인] |
| PnP 의 불확실성 | Ferraz 외(BMVC 2014)는 2D 점마다 다른 가우시안 잡음을 두고 PnP 를 최대우도로 풉니다. Vakhitov 외(CVPR 2021)는 3D 점의 불확실성까지 넣어 KITTI 에서 평균 평행이동 오차를 18% 줄였습니다 | 카메라 위치 추정 기준 | Ferraz [검색], Vakhitov [확인] |
| 잡는 동안의 흔들림 | OAT 의 grasp stabilization (2.1 참고) | 정성적 | [확인] |

**정리 [의견].**

- 손등일 때 떨림이 4배인 원인은 두 가지 가설로 나뉩니다. (가) 손가락이 손바닥을 가려 2D 점이 흔들린다. (나) MediaPipe 가 추정한 손 크기(배율)가 손등 시점에서 더 흔들려 깊이가 흔들린다. (나)라면 2.1 의 배율 고정이 떨림도 함께 줄입니다. 이 둘은 떨림을 깊이 방향과 화면 방향으로 나눠 보면 구별됩니다(6.1).
- 손바닥 법선과 카메라 축의 각도, 또는 PnP 재투영 오차로 필터 세기를 바꾸는 방식은 검색 범위에서 손목 텔레옵 사례를 찾지 못했습니다(Q11, Q12, Q26). 하지만 신뢰도로 관측 잡음을 조절하는 것은 추적 분야에서 흔한 기법이라, 새로움은 작습니다.
- 재투영 오차는 이미 `WristPose.reprojection_px` 로 프레임마다 계산되고 있습니다. 가장자리에서 16 px, 안쪽에서 4~5 px 였으므로(01 문서 V5) 신뢰도 신호로 쓸 수 있는지 볼 만합니다.

### 2.4 모델 선택 (측정 4)

| 모델 | 정확도 | 속도 | 근거 |
|---|---|---|---|
| MediaPipe Hands, Full | 손바닥 크기로 나눈 MSE 13.4%(실제와 합성 데이터 함께 학습) | Pixel 3 에서 16.1 ms. 우리 RTX 5070 Ti 에서 두 손 3.7~4.2 ms | 논문 [확인], 속도 [실측] |
| HaMeR (ViT-H, 671M) | FreiHAND PA-MPJPE 6.0 mm, HO3Dv2 7.7 mm | RTX 4060 Ti 에서 27 FPS (Fast-HaMeR 의 측정) | [확인] |
| Fast-HaMeR 학생 모델 | HO3Dv2 PA-MPJPE 8.1~9.5 mm | 같은 GPU 에서 36~50 FPS | [확인] |
| WiLoR (CVPR 2025) | FreiHAND PA-MPJPE 5.5 mm, HO3D 7.5 mm | 검출기만 RTX 4090 에서 138~175 FPS. 복원까지 포함한 속도는 논문에 없음 | [확인] |
| HandOS | FreiHAND PA-MPJPE 5.0 mm | 확인 못 함 | [확인] |

같은 조건에서 MediaPipe 와 비교한 결과는 다음과 같습니다.

- **WristPP(2026) 부록:** 150도 화각의 머리 착용 카메라에서 MPJPE 가 MediaPipe 46.8 mm, WiLoR 15.0 mm 였습니다. 조명 세 단계 모두 비슷했습니다. MediaPipe 는 손가락이 엉켜 물리적으로 불가능한 3D 손 모양을 내기도 했다고 씁니다 **[확인]**.
- **Manzone 외(2026), HoloLens 2 1인칭 영상:** PA-MPJPE 순서가 WiLoR < HaMeR < HoloLens 2 < MediaPipe < WildHands 였고 모든 차이가 유의했습니다. MediaPipe 는 정답이 있는 프레임의 약 22% 에서 아무것도 내지 못했고, 나머지 세 모델은 1% 미만이었습니다 **[확인]**.
- **OAT:** FreiHAND 정지 영상에서 MediaPipe 가 검출한 비율은 15.33% 였습니다 **[확인]**. 잘린 정지 영상이라 영상 추적과는 조건이 다릅니다 **[의견]**.

**주의할 점 [의견].**

- 위 정확도의 대부분은 PA-MPJPE 입니다. Procrustes 정렬이 위치, 회전, **배율**을 지운 뒤의 오차이므로, 우리 문제인 손목 깊이와 배율 흔들림에 대해서는 직접 말해 주지 않습니다.
- HaMeR 계열의 평행이동은 시간에 따라 흔들리고 떠돈다는 보고가 있고(Large Video Planner [검색]), HaWoR 출력에서도 한 영상 안의 배율이 10 cm 넘게 변했습니다(Dexterous Point Policy [확인]). 무거운 모델로 바꾼다고 배율 흔들림이 사라지지는 않을 가능성이 있습니다.
- 텔레옵 조건(웹캠, 두 손, 손목 떨림, 가장자리 좌우, 지연)에서 MediaPipe, WiLoR, HaMeR 를 같은 영상으로 비교한 결과는 검색 범위에서 찾지 못했습니다(Q13, Q14). 우리 녹화 영상으로 하면 작은 기여가 됩니다.

### 2.5 손이 다시 나타날 때의 튐 (측정 5)

| 접근 | 선행 연구 | 내용 | 근거 |
|---|---|---|---|
| 리타게팅에 매끄러움 항 | AnyTeleop 의 리타게팅 최적화에 `β‖q_t − q_{t−1}‖²` 항이 있습니다. 빠르게 움직이다 추적을 잃으면 멈추고 다시 검출하며, 작업자에게 천천히 움직이라고 안내합니다 | 0.3 s 이상 끊기면 우리 리타게터는 이 연속성을 버립니다 | [확인] |
| 손목에 속도 벌점 | OAT 의 손목 최적화에 이전 프레임과의 위치 차이 벌점(β2)이 있습니다 | 손목 위치에만 | [확인] |
| 다시 나타날 때 보간 | Ferstl 외(SAP 2021)는 VR 아바타에서 추적을 잃으면 마지막 자리에 멈췄다가 돌아올 때 튀는 문제를 다뤘습니다. 세 전략(fade: 멈춘 뒤 돌아오면 관절각을 시간에 걸쳐 보간, rest: 0.4 s 넘게 끊기면 쉬는 자세로, hybrid)이 모두 사람이 느끼는 오류를 줄였습니다. 운동량으로 궤적을 예측하는 방법은 좋지 않았다고 씁니다 | 지각 실험, 로봇 아님 | [확인] |
| 클러치 재고정 | NVIDIA Isaac Teleop 의 SO-101 리타게터는 클러치를 걸 때마다 기준 자세를 다시 잡고, 믿을 만한 프레임이 올 때까지 기준 잡기를 미룹니다. 끊긴 프레임마다 이를 다시 걸어 두는 것이 튐을 막는다고 씁니다 | 03 문서의 클러치에 바로 적용 가능 | [확인] |
| 마지막 명령 유지 | ByteDexter(2025)는 추적기가 다시 나타날 때까지 마지막 명령을 유지해 튐을 막는다는 요약이 있습니다 | | [검색] |
| 저크 제한 궤적 생성 | Ruckig(RSS 2021)는 속도, 가속도, 저크 한계 안에서 임의의 목표 상태로 가는 시간 최적 궤적을 온라인으로 만듭니다. 10억 개가 넘는 무작위 궤적으로 강건성을 시험했습니다 | 목표가 튀어도 명령은 한계 안에 머무릅니다 | [확인] |

**정리 [의견].** 이 항목은 연구가 아니라 공학입니다. 관절 속도 제한, 다시 나타난 뒤 0.3~0.5 s 에 걸친 보간, 팔에 대한 클러치 재고정을 넣으면 됩니다. 새로운 시도의 여지는 없지만, 실물 손과 팔을 움직이기 전에 반드시 필요합니다.

---

## 3. 새로운 시도인가, 알려진 방법의 적용인가

| 후보 | 선행 연구 상태 | 이 프로젝트에서의 위치 |
|---|---|---|
| 1a 작업자별 손 크기 보정(상수) | 손 크기 사전 지식(Iqbal), 작업자별 형상 보정(Kong), 손 기반 카메라 보정(OAT)이 있습니다 | 알려진 방법의 적용 |
| 1b 세션 동안 배율 고정 | HaWoR 출력에 β 고정(Dexterous Point Policy), 자세별 뼈 길이 수렴(Snap 특허)이 있습니다. MediaPipe + PnP 텔레옵에 적용한 사례는 검색 범위에서 찾지 못함 | 알려진 아이디어를 새 구성에 적용 |
| 1c 깊이 카메라 | AnyTeleop, Vuong 외 등 다수 | 알려진 방법. 하드웨어 결정 사항 |
| 1d 단안 깊이 네트워크 | HaWoR, Large Video Planner 등 오프라인 사용 | 알려진 방법. 실시간에는 맞지 않음 |
| 2a 추적 정체성 | SORT 계열 | 알려진 방법의 적용 |
| 2b 3D chirality 교차 확인 | 기하 판정(Snap 특허), 메시 법선(HandOS) | 판정 자체는 알려짐. "MediaPipe 가 라벨을 틀릴 때 3D 모양도 뒤집히는가"는 검색 범위에서 찾지 못함 |
| 3 방향, 불확실성 적응 필터 | 신뢰도 가중 One Euro(포스터), 신뢰도 기반 관측 잡음(MOT), 불확실성 PnP | 손바닥 각도나 재투영 오차로 손목 필터를 조절한 사례는 검색 범위에서 찾지 못함. 새로움은 작음 |
| 4 모델 교체 | 정확도 비교는 많음 | 텔레옵 조건 비교는 검색 범위에서 찾지 못함. 벤치마크 기여 |
| 5 재등장 안전 장치 | Ferstl, Isaac Teleop, Ruckig 등 | 알려진 방법의 적용 |

---

## 4. 순위

기대 효과는 "텔레옵 품질과 안전에 미치는 영향", 노력은 "이 저장소에서 들일 시간과 설치", 새로움은 3장 기준입니다. 모두 **[의견]** 입니다.

| 순위 | 후보 | 기대 효과 | 노력 | 새로움 | 이유 |
|---|---|---|---|---|---|
| 0 (선행 조건) | 5 재등장 안전 장치 | 높음(안전) | 낮음 | 없음 | 실물 전에 반드시 필요합니다. 연구 순위와 따로 둡니다 |
| 1 | 1b 배율 고정 (+1a 상수 보정) | 높음 | 낮음 | 중간 | 녹화 영상만으로 잴 수 있고, 측정 1 과 3 을 함께 겨냥합니다. 클러치 방식에서 해로운 오차를 직접 줄입니다 |
| 2 | 2a + 2b 좌우: 추적 정체성 + chirality | 중간~높음 | 낮음~중간 | 중간 | 지금 가운데 60% 로 묶인 작업 범위를 넓힐 수 있습니다. 어깨 너비로 벌린 손이 x 0.09, 0.86 에 있었으므로(01 문서 V1) 실제 작업에 영향이 큽니다 |
| 3 | 3 불확실성 적응 필터 | 중간 | 낮음 | 낮음~중간 | 1위 실험 결과를 본 뒤에 합니다. 떨림이 배율 때문이면 필요가 줄어듭니다 |
| 4 | 4 모델 교체 벤치마크 | 중간(불확실) | 중간~높음 | 낮음 | 3D 정확도 이득은 크다고 보고되지만, 손목 깊이와 지연에 대한 이득은 모릅니다. 설치가 무겁습니다 |
| 5 | 1c 깊이 카메라 | 높음(절대 위치) | 하드웨어에 달림 | 없음 | 클러치 방식이면 절대 위치의 가치가 줄어듭니다. 01 문서 7장 질문 1(RealSense 유무)의 답이 필요합니다 |
| 6 | 1a 단독 | 낮음 | 매우 낮음 | 없음 | 클러치 방식에서는 상수 배율 오차가 이동 배율 오차로만 나타납니다. 1b 와 함께 하면 비용이 거의 없습니다 |
| 7 | 1d 단안 깊이 네트워크 | 낮음 | 중간 | 없음 | 느리고(91~1300 ms, V100), 장면 단위 정확도도 손 크기 사전 지식보다 낫다고 보기 어렵습니다 |

---

## 5. 추천 실험 두 가지

### 5.1 실험 A: 손목 떨림을 나누고, 배율을 세션 동안 고정하기 (후보 1a, 1b, 3)

**가설.** 손등일 때 떨림의 상당 부분은 깊이 방향이고, 그 원인은 MediaPipe world landmark 의 프레임별 배율 흔들림이다.

**단계.**

1. **로봇 없이, 녹화 영상으로.** `data/clips/v2_palm2.avi`(손바닥, 안쪽), `v2_back.avi`(손등, 안쪽), `v2_cross.avi` 를 다시 처리합니다. 프레임마다 다음을 기록합니다.
   - 손바닥 배율 `s_t`: world landmark 에서 손목(0)부터 MCP(5, 9, 13, 17)까지 거리의 평균. 손허리뼈 길이라 손을 쥐어도 거의 변하지 않아야 하는 값입니다.
   - PnP 로 구한 손목 위치를 카메라 z(깊이)와 x, y(화면 방향)로 나눈 값.
   - 재투영 오차, 손바닥 법선과 카메라 축 사이 각도.
2. **떨림 나누기.** 지금과 같은 정의(2차 차분 중앙값)로 깊이 방향과 화면 방향 떨림을 따로 구합니다. `s_t` 의 상대 변동(2차 차분 / 중앙값)과 깊이 떨림의 상관을 봅니다.
3. **배율 고정.** PnP 에 넣기 전에 world landmark 를 `s̄ / s_t` 배 합니다. `s̄` 는 세션 처음 N 초의 중앙값입니다. 같은 영상으로 떨림을 다시 잽니다.
4. **손을 쥐었다 펴기 (새 녹화, 카메라만).** 손목을 책상에 대고 고정한 채 손을 5회 쥐었다 폅니다. 손바닥과 손등 각각 녹화합니다. 손목은 움직이지 않았으므로 PnP 깊이의 변화가 곧 오차입니다. 고정 전후를 비교합니다.
5. **작업자 보정(1a).** 줄자로 작업자의 손목에서 중지 MCP 까지 길이를 재고, `s̄` 를 그 값에 맞춥니다. 01 문서 V4(40, 60, 80 cm)를 보정 전후로 측정합니다. 카메라 내부 파라미터 보정이 먼저 필요합니다.

**지표.**

| 지표 | 정의 | 판단 기준(초안) |
|---|---|---|
| 깊이 떨림 비율 | 깊이 방향 떨림 / 전체 떨림, 손바닥과 손등 각각 | 손등에서 깊이 비율이 크면 가설 지지 |
| 배율과 깊이의 상관 | `s_t` 변동과 깊이 떨림의 상관계수 | |
| 고정 후 손목 떨림 | 손등 영상의 손목 떨림 mm, 고정 전후 | 의미 있게 줄면 기본값으로 채택. 줄어드는 폭의 목표는 1단계 결과를 보고 정합니다 |
| 쥐었다 펼 때의 가짜 깊이 이동 | 손목 고정 상태에서 PnP 깊이의 최대 변화 mm | |
| V4 거리 오차 | 40, 60, 80 cm 에서 평균 오차와 표준편차 | 보정 후 배율 오차가 거리에 무관하게 일정한지 |
| 처리 시간 | 프레임당 추가 시간 | 무시할 만해야 함(점 21개 배율 조정뿐) |

**주의.** 배율을 고정하면 MediaPipe 가 배율을 맞게 바꾼 경우(예: 손이 실제로 기울어 원근이 바뀔 때)까지 막을 수 있습니다. 손허리뼈 길이를 쓰는 이유가 이것이지만, 손등 시점에서 MCP 점 자체가 흔들린다면 효과가 줄어듭니다. 그 경우 OAT 의 grasp stabilization 처럼 "손 모양이 빠르게 바뀌는 동안 깊이 갱신을 늦추는" 방식을 함께 시험합니다.

### 5.2 실험 B: 가장자리 좌우 라벨을 추적 정체성과 chirality 로 정하기 (후보 2a, 2b)

**질문.** (1) 지금 규칙을 거친 최종 배정은 가장자리에서 얼마나 틀리는가. (2) 라벨이 틀린 프레임에서 MediaPipe 의 3D 손 모양도 뒤집혀 있는가. (3) 트랙 정체성 규칙으로 최종 배정 오류를 얼마나 줄이는가.

**단계.**

1. **최종 배정 오류 세기.** `v2_palm.avi`(가장자리까지)를 지금 노드 로직으로 다시 처리하고, 위치 관계(영상 왼쪽 손이 오른손)를 정답으로 최종 배정 오류율을 셉니다. 영역별(x < 0.2, 0.2~0.8, x > 0.8)로 나눕니다. 지금 jsonl 에는 랜드마크가 없으므로 영상을 다시 처리해야 합니다.
2. **chirality 와 라벨 비교.** 같은 프레임에서 리타게터와 같은 방식의 chirality 값을 계산합니다. 2×2 표(라벨 맞음/틀림 × chirality 맞음/틀림)를 만듭니다. 라벨이 틀린 프레임에서 chirality 가 맞는 비율이 핵심 결과입니다. 손등 영상과 손바닥 영상을 모두 봅니다.
3. **트랙 정체성 규칙.** 두 손을 손목 영상 좌표로 프레임 사이에 짝짓습니다(두 개뿐이므로 헝가리안 대신 두 가지 짝 중 거리 합이 작은 쪽). 트랙이 시작될 때 좌우를 정하고, 이후에는 라벨, chirality, 위치를 프레임마다 투표에 넣되 가운데 영역의 표에 가중치를 더 줍니다. 0.3 s 넘게 끊기면 트랙을 끝냅니다.
4. **새 녹화.** 손을 가장자리 밖으로 뺐다 다시 넣기, 한 손만 가장자리에 두기, 교차를 각 15초씩 녹화합니다. 교차 영상은 정답이 없으므로 손목 연속성을 정답 대용으로 쓰고, 그 한계를 함께 적습니다.

**지표.**

| 지표 | 정의 |
|---|---|
| 최종 배정 오류율 | 영역별, 규칙별(지금 규칙 / 트랙 규칙 / 트랙 + chirality) |
| chirality 의 독립성 | 라벨이 틀린 프레임 중 chirality 가 맞는 비율 |
| 좌우가 뒤바뀐 횟수 | 영상당, 교차 영상 포함 |
| 사용 가능한 가로 폭 | 최종 배정 오류가 0 인 x 범위 |
| 처리 시간 | 프레임당 추가 시간 |

**판단.** chirality 가 라벨 오류와 독립적이면, 가장자리의 단일 손도 바르게 배정할 수 있고 운용 조건(가운데 60%)을 풀 수 있습니다. 같이 뒤집혀 있다면 트랙 정체성만으로 가며, 손이 가장자리에서 **처음** 나타나는 경우는 여전히 위험하므로 운용 조건을 유지합니다.

### 5.3 실물 전 필수 항목: 재등장 안전 장치 (후보 5)

연구 순위와 별도로, 실물 손과 팔을 움직이기 전에 다음을 시뮬레이션에서 확인합니다.

1. **오프라인 재현.** `v2_cross.avi` 의 0.2~1.0 s 끊김 10개(02 문서)에 더해, 녹화 영상에서 0.3, 0.5, 1, 2 s 구간을 인위로 지운 영상을 만듭니다. 리타게터와 03 문서의 IK 를 오프라인으로 돌립니다.
2. **비교할 조건.** (a) 지금 그대로, (b) 관절별 속도 제한(Allegro 와 xArm7 사양의 한계보다 낮게), (c) 다시 나타난 뒤 0.3~0.5 s 에 걸친 보간(Ferstl 의 fade), (d) 팔은 끊긴 뒤 클러치 기준 재고정(Isaac Teleop 방식), (e) Ruckig 로 저크 제한.
3. **지표.** 재등장 직후 최대 관절 속도와 그 한계 대비 비율, 한계를 넘은 프레임 수, 목표에 수렴하기까지 시간, 정상 구간에서 더해진 지연(보간이 평소 동작을 늦추지 않는지).
4. **실물.** 시뮬레이션에서 고른 조건으로 손 한 쪽부터, 팔은 낮은 속도 상한으로 시험합니다. 실물을 움직이기 전에 미리 말씀드립니다.

Ruckig 는 새 패키지라서 쓰기로 하면 설치 목록을 먼저 보여 드리겠습니다. (b)와 (c)만으로 충분하면 설치하지 않습니다.

---

## 6. 나머지 후보의 실험 (요약)

- **불확실성 적응 필터(3).** 실험 A 의 기록으로 재투영 오차, 손바닥 각도와 손목 오차의 관계를 먼저 봅니다. 관계가 뚜렷하면 One Euro 의 `min_cutoff` 를 그 값으로 낮추는 변형과, 관측 잡음을 그 값으로 키우는 칼만 필터를 같은 영상에서 비교합니다. 지표는 손바닥과 손등 각각의 떨림과, 빠른 동작에서의 지연(필터 출력과 원 신호의 시간차)입니다.
- **모델 교체(4).** WiLoR 를 녹화 영상에 오프라인으로 돌려, 실험 A 와 B 의 지표(깊이 떨림, 가장자리 배정 오류, 처리 시간)를 MediaPipe 와 같은 표로 비교합니다. WiLoR 는 PyTorch 와 모델 가중치가 필요하므로 설치 목록을 먼저 보여 드립니다. 별도 가상환경에 둡니다.
- **깊이 카메라(1c).** 01 문서 7장 질문 1 의 답을 받은 뒤 정합니다. 있다면 같은 V4 측정을 깊이 카메라 값과 PnP 값으로 나란히 재는 것이 가장 직접적인 비교입니다.

---

## 7. 확인하지 못한 것

- HandOS 의 좌우 정확도(HInt 97.9%, ViTPose 93.8%)는 검색 요약에서만 봤고, 보충 자료가 열리지 않아 원문에서 확인하지 못했습니다.
- WiLoR 의 검출기와 복원을 합친 전체 속도는 논문에 없습니다. HaMeR 의 27 FPS 는 Fast-HaMeR 저자가 RTX 4060 Ti 에서 잰 값입니다.
- Fast-HaMeR 초록의 "1.5배 빠르고 정확도 차이 0.4 mm"는 본문 표(최선 학생 8.1 mm 대 HaMeR 7.7 mm, 1.48배)와 맞습니다. 다른 학생 모델의 증류 후 수치는 표 전체를 보지 않았습니다.
- HandDGP 의 카메라 좌표 오차 수치, Kong 외(ECCV 2022)의 형상 보정 이득 수치는 초록에 없어 확인하지 못했습니다.
- Ferraz 외(BMVC 2014), Martini 외(2024), ByteDexter, Large Video Planner, 1인칭 위치 기반 좌우 판정 연구는 검색 요약만 봤습니다.
- Snap 의 두 특허 공개는 요약 페이지(nweon)로 내용을 봤고, USPTO 원문은 열지 않았습니다. 특허라 성능 수치가 없습니다.
- 기존 조사 문서가 AnyTeleop 의 카메라 비교를 Table V 로 적었는데, 이번에 본 arXiv HTML 판에서는 Table IV 였고 오류율의 정의(틀린 건반 수 / 전체 건반 수)도 확인했습니다. 판에 따라 표 번호가 다를 수 있습니다.
- 우리 쪽 미확인: 01 문서 V5 의 떨림 값이 One Euro 필터 전인지 후인지 이 문서에서는 다시 확인하지 않았습니다. 실험 A 에서는 필터 전 값으로 통일합니다.

---

## 참고문헌

| 문헌 | 링크 | 확인 수준 |
|---|---|---|
| Zhang 외, MediaPipe Hands: On-device Real-time Hand Tracking, 2020 | [arXiv:2006.10214](https://arxiv.org/abs/2006.10214) | 본문 1~3쪽 확인 |
| MediaPipe Hands 문서 (handedness, world landmark, 추적) | [GitHub docs](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md) | 확인 |
| Iqbal 외, Hand Pose Estimation via Latent 2.5D Heatmap Regression, ECCV 2018 | [arXiv:1804.09534](https://arxiv.org/abs/1804.09534) | 초록 확인 |
| Qin 외, AnyTeleop, RSS 2023 | [arXiv:2307.04577](https://arxiv.org/abs/2307.04577) | HTML 본문 확인(Table IV, 8장, 식 1) |
| Vuong 외, Single RGB-D Camera Teleoperation for General Robotic Manipulation, 2021 | [arXiv:2106.14396](https://arxiv.org/abs/2106.14396) | HTML 본문 5.3 확인 |
| Sivakumar, Shaw, Pathak, Robotic Telekinesis, RSS 2022 | [arXiv:2202.10448](https://arxiv.org/abs/2202.10448) | 초록 확인 |
| Chen, Chen, Yao, Dogar, Pose Retargeting from a Single RGB Camera (OAT), ICRA 2026 | [White Rose eprint 238695](https://eprints.whiterose.ac.uk/id/eprint/238695) | 승인본 PDF 전체 확인. arXiv 번호 없음 |
| Kong 외, Identity-Aware Hand Mesh Estimation and Personalization from RGB Images, ECCV 2022 | [arXiv:2209.10840](https://arxiv.org/abs/2209.10840) | 초록 확인 |
| Kim 외, Dexterous Point Policy, 2026 | [arXiv:2606.10614](https://arxiv.org/abs/2606.10614) | HTML 부록 F 확인 |
| Zhang 외, HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos, CVPR 2025 | [arXiv:2501.02973](https://arxiv.org/abs/2501.02973) | HTML 본문 확인(Metric3D, 실시간 아님) |
| Large Video Planner Enables Generalizable Robot Control, 2025 | [arXiv:2512.15840](https://arxiv.org/abs/2512.15840) | 검색 요약 |
| Millerdurai 외, EgoForce, 2026 | [arXiv:2605.12498](https://arxiv.org/abs/2605.12498) | 초록 확인 |
| Valassakis, Garcia-Hernando, HandDGP, ECCV 2024 | [arXiv:2407.15844](https://arxiv.org/abs/2407.15844) | 초록 확인, 수치 미확인 |
| Jing 외, ScaleHP, 2026 | [arXiv:2606.25619](https://arxiv.org/abs/2606.25619) | 초록 확인 |
| Ren 외, Estimating Accurate Hand Pose in Camera Space with Vision Transformer, 2026 | [arXiv:2609.24424](https://arxiv.org/abs/2609.24424) | 초록 확인 |
| Bochkovskii 외, Depth Pro, 2024 | [arXiv:2410.02073](https://arxiv.org/abs/2410.02073) | HTML 표 1, 표 5 확인 |
| Snap Inc., Hand pose-dependent scale estimation for extended reality, US 20260073545 | [nweon 요약](https://patent.nweon.com/43234) | 특허 요약 페이지 확인 |
| Snap Inc., Hand chirality estimation for extended reality tracking, US 20260064192 | [nweon 요약](https://patent.nweon.com/43157) | 특허 요약 페이지 확인 |
| Chen 외, HandOS: 3D Hand Reconstruction in One Stage, CVPR 2025 | [arXiv:2412.01537](https://arxiv.org/abs/2412.01537) | 초록 확인. 좌우 정확도는 검색 요약 |
| Bewley 외, Simple Online and Realtime Tracking (SORT), ICIP 2016 | [arXiv:1602.00763](https://arxiv.org/abs/1602.00763) | 초록 확인 |
| TouchDesigner 포럼, 두 손 번호가 뒤바뀌는 보고 | [forum.derivative.ca](https://forum.derivative.ca/t/making-sure-that-hand-1-is-always-left-hand-with-mediapipe/990357) | 검색 요약 |
| Egocentric Hand Track and Object-based Human Action Recognition, 2019 | [arXiv:1905.00742](https://arxiv.org/abs/1905.00742) | 검색 요약 |
| The Surprising Effectiveness of Video Diffusion Models for Hand Motion Reconstruction, 2026 | [arXiv:2606.30308](https://arxiv.org/abs/2606.30308) | 검색 요약 |
| Casiez, Roussel, Vogel, 1€ Filter, CHI 2012 (DOI 10.1145/2207676.2208639) | [저자 페이지](https://gery.casiez.net/1euro/) | 저자 페이지 확인 |
| Adzhiev, Hosken, Reducing motion jitter in monocular motion capture data, SCA 2026 포스터 | [Bournemouth eprint 42215](https://eprints.bournemouth.ac.uk/42215/) | 초록 확인, 수치 없음 |
| Martini 외, A Robust Filter for Marker-less Multi-person Tracking in HRI Scenarios, 2024 | [arXiv:2406.01832](https://arxiv.org/abs/2406.01832) | 검색 요약 |
| Ferraz, Binefa, Moreno-Noguer, Leveraging Feature Uncertainty in the PnP Problem, BMVC 2014 | [BMVC 페이지](https://bmva-archive.org.uk/bmvc/2014/papers/paper065/index.html) | 검색 요약 |
| Vakhitov 외, Uncertainty-Aware Camera Pose Estimation from Points and Lines, CVPR 2021 | [arXiv:2107.03890](https://arxiv.org/abs/2107.03890) | 초록 확인 |
| Pavlakos 외, Reconstructing Hands in 3D with Transformers (HaMeR), CVPR 2024 | [arXiv:2312.05251](https://arxiv.org/abs/2312.05251) | 초록 확인. 학회는 검색 요약 |
| Potamias 외, WiLoR, CVPR 2025 | [arXiv:2409.12259](https://arxiv.org/abs/2409.12259) | HTML 표 1, 3, 4 확인 |
| Jillani 외, Fast-HaMeR, 2026 | [arXiv:2603.16444](https://arxiv.org/abs/2603.16444) | HTML 표 1, 2, 4 확인 |
| Xi 외, WristPP, 2026 | [arXiv:2603.00606](https://arxiv.org/abs/2603.00606) | HTML 부록 표 10 확인 |
| Manzone 외, Impact of Hand Impairment and Occlusions on Hand Pose Estimation Accuracy in AR, 2026 | [arXiv:2606.17427](https://arxiv.org/abs/2606.17427) | HTML 본문 확인 |
| Ferstl, McDonnell, Neff, Evaluating Study Design and Strategies for Mitigating the Impact of Hand Tracking Loss, SAP 2021 | [저자 PDF](https://www.cs.ucdavis.edu/~neff/papers/HandTrackingLossAndStudyDesign_SAP21.pdf), DOI 10.1145/3474451.3476235 | 본문 1~4쪽 확인 |
| NVIDIA Isaac Teleop, SO-101 retargeter 문서 | [문서 페이지](https://nvidia.github.io/IsaacCapture/main/references/retargeting/so101.html) | 확인 |
| Wen 외, Dexterous Teleoperation of 20-DoF ByteDexter Hand via Human Motion Retargeting, 2025 | [arXiv:2507.03227](https://arxiv.org/abs/2507.03227) | 초록 확인. 추적 끊김 처리는 검색 요약 |
| Berscheid, Kröger, Jerk-limited Real-time Trajectory Generation with Arbitrary Target States (Ruckig), RSS 2021 | [arXiv:2105.04830](https://arxiv.org/abs/2105.04830) | 초록 확인 |

이 저장소의 근거: `docs/01_vision_design.md`(V1, V2, V5), `docs/02_bimanual_hand_sim.md`(3, 4장), `docs/03_arm_follow_design.md`(2장 클러치), `vision/hand_tracker.py`(`RESET_GAP_S`, 필터 설정).

---

## 부록 A. 검색어

2026-10-19 에 웹 검색 도구로 실행했습니다. 번호는 본문에서 "검색 범위에서 찾지 못함"의 근거로 씁니다.

| 번호 | 검색어 | 주제 |
|---|---|---|
| Q1 | monocular RGB hand absolute depth camera-space hand mesh recovery root depth estimation | 1 |
| Q2 | HandDGP camera-space hand mesh prediction differentiable global positioning ECCV 2024 | 1 |
| Q3 | ScaleHP scale-mediated metric-space hand pose estimation arXiv 2606.25619 | 1 |
| Q4 | personalized hand shape calibration MANO beta per user improves absolute 3D hand position monocular | 1 |
| Q5 | Depth Pro Depth Anything V2 metric UniDepth Metric3D v2 inference latency zero-shot metric depth accuracy comparison | 1 |
| Q6 | single RGB camera teleoperation dexterous hand wrist position monocular Robotic Telekinesis HaMeR teleop arm | 1 |
| Q7 | MediaPipe handedness misclassification image border left right hand swap tracking | 2 |
| Q8 | hand chirality classification from 3D keypoints left right hand determination geometric | 2 |
| Q9 | multi-hand tracking identity association Hungarian Kalman two hands occlusion crossing egocentric | 2 |
| Q10 | uncertainty-aware hand pose filtering Kalman measurement covariance self-occlusion hand tracking teleoperation | 3 |
| Q11 | confidence-adaptive Kalman filter keypoint measurement noise scaled by detection confidence pose estimation video smoothing | 3 |
| Q12 | One Euro filter variant confidence weighted adaptive jitter latency hand tracking keypoints improved 1€ filter | 3 |
| Q13 | WiLoR HaMeR inference speed FPS comparison FreiHAND PA-MPJPE | 4 |
| Q14 | MediaPipe Hands vs WiLoR HaMeR benchmark accuracy comparison 3D hand pose teleoperation evaluation | 4 |
| Q15 | WiLoR real-time teleoperation Hz latency hand pose estimator robot teleop uses WiLoR | 4 |
| Q16 | vision-based teleoperation hand tracking loss reacquisition jump safety rate limit dexterous | 5 |
| Q17 | Ruckig online jerk-limited trajectory generation teleoperation real-time target jumps | 5 |
| Q18 | teleoperation clutch re-engagement smooth transition target blending avoid discontinuity robot arm hand tracking lost | 5 |
| Q19 | VR hand tracking reacquisition smoothing pose jump filter after tracking lost Quest hand tracking latency jitter study | 5 |
| Q20 | Ferstl McDonnell Neff 2021 hand tracking loss social VR amelioration strategies | 5 |
| Q21 | HaWoR world-space hand motion reconstruction egocentric CVPR 2025 Dyn-HaMR scale drift | 1 |
| Q22 | handedness estimation video temporal consistency left right hand label flipping palm dorsal ambiguity | 2 |
| Q23 | "handedness" accuracy benchmark hand detector left right classification near image boundary truncated hands | 2 |
| Q24 | online estimation of hand scale over time monocular hand tracking depth from apparent hand size Kalman filter scale state | 1 |
| Q25 | MediaPipe handedness wrong label world landmarks mirrored 3D chirality inconsistent issue github | 2 |
| Q26 | orientation-dependent measurement noise filter hand tracking palm facing away camera adaptive smoothing wrist pose | 3 |
| Q27 | hand pose estimation error depends on viewpoint dorsal view vs palmar view accuracy monocular RGB | 3 |
| Q28 | "hand size" calibration MediaPipe teleoperation operator hand length scale wrist depth correction | 1 |
| Q29 | monocular metric depth network combined with hand pose to resolve scale ambiguity wrist translation robot learning from human video | 1 |
| Q30 | PnP pose uncertainty covariance propagation keypoint noise Ferraz "Leveraging feature uncertainty in the PnP problem" | 3 |
| Q31 | Vakhitov uncertainty-aware camera pose estimation points lines CVPR 2021 EPnPU DLSU | 3 |
| Q32 | MediaPipe hand landmarker world landmarks hand size metric scale assumption average hand depth error | 1 |
| Q33 | Iqbal hand pose estimation latent 2.5D heatmap regression absolute depth recovery known bone length scale normalization | 1 |
| Q34 | two-hand tracking identity left right assignment video tracker hand side classification temporal smoothing egocentric dataset method | 2 |
| Q35 | DexPilot vision-based teleoperation dexterous robotic hand-arm system Handa ICRA 2020 arXiv | 배경 |
| Q36 | Dexterous Point Policy scale-consistent HaWoR hand scale varies 10 cm egocentric video arXiv 2606.10614 | 1 |
| Q37 | Large Video Planner HaMeR predicted translations noisy drift MegaSaM wrist trajectory metric arXiv 2512.15840 | 1 |
| Q38 | arXiv 2406.01832 hand tracking filter human-robot interaction Kalman OpenPose motion capture | 3 |
| Q39 | Casiez Roussel Vogel 1€ filter CHI 2012 paper hal | 3 |

검색 결과에는 오래된 자료와 새 자료가 섞여 있으며, 검색 도구가 미국 지역 결과만 돌려준다는 제약이 있습니다. 학회 논문집 내부 검색(IEEE Xplore, ACM DL)은 하지 않았습니다.
