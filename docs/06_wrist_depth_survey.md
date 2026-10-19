# 손목 깊이 흔들림 조사: 깊이 카메라, 스테레오 IR, 광선 방향 필터

- 작성: 2026-10-19, Claude subagent (Claude Opus 5.5), 문헌 조사 담당
- 근거 문서: `docs/05_ego_camera.md`, `docs/results/r4_ego_right_wrist_follow.md`, `docs/results/m2_scale_jitter.md`, `docs/04_rnd_candidates_survey.md`
- 이 문서는 조사 결과이며, 새로 잰 값은 없습니다. 카메라와 로봇은 쓰지 않았습니다.

**표기.** 04 문서와 같습니다.

| 표기 | 뜻 |
|---|---|
| **[확인]** | 논문 초록 페이지나 본문(HTML, PDF)을 직접 열어 확인했습니다 |
| **[검색]** | 검색 결과의 요약만 봤습니다. 인용하기 전에 원문을 확인해야 합니다 |
| **[실측]** | 이 저장소에서 잰 값입니다 |
| **[의견]** | 조사와 측정을 바탕으로 한 판단이나 계산입니다 |

"새로운 시도인지"는 "검색 범위에서 찾지 못함"으로만 쓰고, 그때 쓴 검색어 번호(W1~W34, 부록 A)를 붙였습니다. 찾지 못했다는 것이 없다는 뜻은 아닙니다.

---

## 요약

- **가장 큰 이득은 깊이를 센서에서 직접 받는 것입니다.** 이것 자체는 새롭지 않습니다(GMH-D, AnyTeleop, Vuong 외, DexMV, Sensors 2026 논문). 다만 USB 3 케이블을 기다리지 않고 시험할 길이 하나 있습니다. **컬러 대신 왼쪽 IR 영상(Y8)에 MediaPipe 를 돌리고 깊이를 같은 장치에서 받는 방식**입니다. 깊이 영상은 왼쪽 IR 카메라 좌표에서 만들어지므로 정렬이 필요 없고, Y8 은 컬러(YUYV)보다 대역폭이 작습니다 **[의견]**. USB 2 에서 이 조합이 몇 fps 로 도는지, MediaPipe 가 프로젝터 점무늬가 있는 IR 영상에서 손을 잘 잡는지가 첫 측정입니다.
- **광선 방향 필터는 오늘 바로 할 수 있는 보조 수단입니다.** 지금 One Euro 필터는 카메라 좌표 x, y, z 성분마다 같은 설정을 쓰고, 깊이 잡음 자체가 속도로 읽혀 차단 주파수를 올릴 수 있습니다 **[의견]**. 깊이 성분만 따로 세게 거르는 것은 새롭지 않습니다. RealSense 와 MediaPipe 로 팔을 추적하는 텔레옵 논문이 칼만 필터 관측 잡음을 `σz ≫ σx, σy` 로 둡니다(Kwok 외 2026) **[확인]**.
- **스테레오 IR 키포인트 삼각측량은 기대보다 이득이 작을 수 있습니다.** D457 기선 95 mm, 거리 0.6 m 에서 시차 1 px 오차는 깊이 약 8.5 mm(848x480) 또는 15 mm(480x270, USB 2 해상도)에 해당합니다 **[의견, 계산]**. MediaPipe 키포인트의 시차 잡음이 1 px 를 넘으면 지금 PnP 보다 크게 낫지 않고, 조밀 깊이 영상을 쓰는 쪽이 훨씬 정밀합니다. 깊이 구멍과 손 가장자리 문제를 피하는 대안으로만 남깁니다.
- **깊이 카메라를 선생으로 한 작업자별 깊이 보정기는 새로움이 가장 큽니다.** 깊이 영상을 학습에만 쓰고 RGB 로 추론하는 연구는 많지만(Cai 외 2018, Yuan 외 2018, Zhao 외 2020), 모두 큰 네트워크를 데이터셋으로 학습합니다. "MediaPipe 특징 몇 개에서 작업자별 작은 회귀기로 미터 깊이를 맞춘다"는 형태는 검색 범위에서 찾지 못했습니다(W1, W4~W6, W8, W9). 하지만 이 방식은 고정 편향(손바닥과 손등의 9% 배율 차이 같은 것)은 줄여도 **프레임마다의 흔들림은 줄인다는 보장이 없습니다.** 입력이 같은 MediaPipe 출력이기 때문입니다.

---

## 1. 문제 다시 보기

| | 측정 | 출처 |
|---|---|---|
| 1 | 에고 RealSense 에서 오른손 손목 프레임 간 이동 p95 가 로봇 앞, 위 축에서 42.0, 43.4 mm, 좌우 축에서 8.6 mm 였습니다. 카메라가 아래로 44° 기울어 깊이 오차가 앞, 위 축에 섞입니다 | R4 **[실측]** |
| 2 | 녹화 영상의 정지 구간에서 깊이 방향 2차 차분이 화면 방향의 2~8배였습니다. 손등에서는 상대 깊이 변화가 MediaPipe 배율 변화를 기울기 +0.71~+1.01 로 따라갔고, 손바닥에서는 따라가지 않았습니다 | M2 **[실측]** |
| 3 | 배율 중앙값이 손등 88.6~89.4 mm, 손바닥 97.0~98.2 mm 로 약 9% 달랐습니다(따로 찍은 영상) | M2 **[실측]** |
| 4 | USB 2 에서 컬러 + 깊이 15 fps 요청이 11.2 fps 로 돌았고, 깊이 480x270 단독은 27.7 fps 였습니다 | 05 문서 4장 **[실측]** |
| 5 | 위치 필터는 카메라 좌표 성분마다 같은 One Euro(min_cutoff 1.0 Hz, beta 10.0)입니다 | `vision/hand_tracker.py`, `vision/filters.py` |

R4 의 p95 가 필터 전 값인지 후 값인지는 R4 보고서에 적혀 있지 않습니다. 아래 비교에서는 크기의 감만 잡는 데 씁니다.

**D457 에 대해 알아 둘 것.**

| 항목 | 값 | 근거 |
|---|---|---|
| 기선 | 95 mm | FRAMOS 제품 페이지 **[확인]** |
| 최소 깊이 거리(MinZ) | 520 mm (최대 해상도 기준), 권장 범위 0.6~6 m | FRAMOS **[확인]** |
| MinZ 와 해상도 | MinZ 는 출력 가로 해상도에 비례해 줄어듭니다 | Intel 튜닝 자료 **[검색]** |
| 깊이 RMS 오차 | `거리² × subpixel / (초점거리 px × 기선)`. 질감이 좋은 평면에서 subpixel 0.1 미만, 0.05 근처까지 나온다고 씁니다 | Intel 자가 보정 문서가 같은 변수를 정의 **[확인]**, 식과 subpixel 값은 Intel 튜닝 자료 **[검색]** |
| 깊이 시야각 | 86°(FRAMOS 설명) 또는 87° x 58°(Rutronik 브로셔). 자료끼리 다릅니다 | **[확인]** FRAMOS, **[검색]** Rutronik |
| 주 인터페이스 | GMSL/FAKRA. "USB 단자는 생산과 디버그용"이라는 Intel 보도자료 요약이 있습니다 | **[검색]** (Intel 페이지가 열리지 않음) |
| USB 2 모드 | 깊이와 IR 480x270, 424x240 에서 6/15/30 Hz 가 열거된다는 사용자 보고 | **[검색]** |

**시차 1 px 의 깊이값 [의견, 계산].** HFOV 87° 로 가정하면 초점거리는 가로 1280 에서 약 674 px, 848 에서 447 px, 480 에서 253 px 입니다. 거리 z = 0.6 m, 기선 0.095 m 에서 `z² / (f·B)` 는 각각 5.6, 8.5, 15.0 mm/px 입니다. subpixel 0.08 이면 조밀 깊이의 잡음은 0.5~1.2 mm 수준이고, 키포인트 시차가 1 px 흔들리면 5.6~15 mm 입니다. 손 피부와 가장자리에서는 평면 값보다 나빠지므로, 이 계산은 하한으로만 읽어야 합니다.

**작업 거리와 MinZ.** 에고 카메라에서 손까지의 절대 거리는 아직 재지 않았습니다. 0.52 m 보다 가까우면 1280x720 깊이에서는 손이 MinZ 안에 들어갑니다. 848x480 이나 480x270 을 쓰면 MinZ 가 줄어듭니다 **[검색 근거의 의견]**.

---

## 2. 후보별 선행 연구

### 2.1 깊이 카메라를 선생으로 한 단안 손목 깊이 보정

**생각.** RealSense 깊이로 손바닥 깊이를 정답 삼아, MediaPipe 특징(world landmark 배율, 이미지에서 본 손 크기, 손바닥 법선과 광선의 각도, 화면 위치, 재투영 오차 등)에서 미터 깊이로 가는 작은 회귀기를 작업자마다 학습합니다. 나중에 깊이가 없는 카메라에서 이 회귀기로 깊이를 고칩니다.

| 접근 | 선행 연구 | 보고된 효과 | 근거 |
|---|---|---|---|
| 학습 때만 깊이, 추론은 RGB | Cai 외(ECCV 2018)는 예측한 3D 자세에서 깊이 지도를 그려 실제 RGB-D 깊이와 비교하는 "depth regularizer"를 약한 감독으로 씁니다. 시험 때는 RGB 만 씁니다 | 약지도와 완전지도 모두 개선(초록에 수치 없음) | [확인] 초록 |
| 깊이 네트워크 특징을 흉내 | Yuan, Stenger, Kim(2018)은 짝지은 RGB-D 를 특권 정보로 써서, RGB 네트워크의 중간 특징이 깊이 네트워크 특징을 닮도록 학습합니다. 시험 때는 RGB 만 씁니다 | 세 데이터셋에서 RGB 최고 수준을 넘었다고 씀(초록에 수치 없음) | [확인] 초록 |
| 깊이 선생 → RGB 학생 증류 | Zhao 외(CVPR 2020)는 깊이 등 우월한 입력으로 학습한 선생의 지식을, 선생이 없는 데이터셋의 RGB 학생에 사전분포로 옮깁니다 | "경쟁력 있는 성능"(초록에 수치 없음) | [확인] 초록 |
| 자기지도 시험 시 최적화 | V-HPOT(Mucha 외, WACV 2026)는 초점거리와 영상 크기로 정규화한 가상 카메라 공간에서 깊이를 예측하고, 추론 중에 예측과 배율을 바꾼 예측 사이의 3D 일관성 손실로 깊이 인식을 다듬습니다. 정답 라벨은 쓰지 않습니다 | 영역 밖 평가에서 평균 자세 오차 H2O 71%, AssemblyHands 41% 감소 | [확인] 초록 |
| 깊이로 손 모델 개인화 | Remelli 외(ICCV 2017)는 깊이 측정 모음으로 작업자의 손 추적 모델을 개인화합니다 | 확인 못 함 | [검색] |
| RGB 몇 장으로 형상 개인화 | Kong 외(ECCV 2022), 04 문서 2.1 | 초록에 수치 없음 | [확인] (04 문서) |
| 자세별 손 크기 수렴(특허) | Snap 특허(US 20260073545), 04 문서 2.1 | 특허라 수치 없음 | [확인, 요약 페이지] (04 문서) |
| 깊이 선생 → RGB 정책 | MonoLift(NeurIPS 2025)는 깊이를 쓰는 선생에서 RGB 만 쓰는 조작 정책으로 증류합니다. 손 자세가 아니라 정책입니다 | 확인 못 함 | [검색] |
| 깊이 센서 잡음과 회귀 | Lienen 외(ACML 2021)는 흔한 RGB-D 센서의 깊이 잡음이 최소제곱 회귀의 통계 가정을 깬다고 보고하고 강건 회귀를 씁니다 | 라벨로 쓸 깊이의 잡음 처리 문제 | [검색] |

**정리 [의견].**

- 깊이를 학습 신호로 쓰고 RGB 로 추론하는 큰 틀은 2018년부터 있습니다. 모두 데이터셋 단위로 큰 네트워크를 학습하는 연구입니다.
- "MediaPipe 출력 위에 작업자별, 설치별 작은 회귀기를 RealSense 깊이로 학습"하는 형태는 검색 범위에서 찾지 못했습니다(W1, W4, W5, W6, W8, W9). 시험 시 적응(W2, W3)도 손에 대해서는 V-HPOT 정도만 찾았고, 그것은 라벨 없는 일관성 손실입니다.
- **무엇을 고칠 수 있고 무엇을 못 고치는지 구분해야 합니다.** 회귀기의 입력은 흔들리는 MediaPipe 출력이므로, 입력 잡음이 정답과 무관하게 흔들리면 출력도 흔들립니다. M2 에서 이미지에서 본 손 크기와 world 배율의 상관이 -0.19~+0.35 로 약했다는 점(M2 7장)은, 이미지 쪽 크기가 world 배율과 다른 정보를 담고 있을 가능성을 남깁니다. 그 정보가 깊이 정답과 더 맞는지가 이 후보의 성패를 가릅니다. 반대로 고정 편향, 예를 들어 손바닥과 손등 사이의 9% 배율 차이나 작업자 손 크기는 회귀기가 손바닥 법선 각도를 입력으로 받으면 잘 고칠 수 있는 종류입니다.
- 라벨 쪽 위험도 있습니다. 손목 점 하나의 깊이는 손 가장자리와 배경에 걸리기 쉽습니다. 손바닥 다섯 점(0, 5, 9, 13, 17)으로 만든 다각형 안의 유효 깊이 중앙값을 라벨로 쓰는 편이 안전합니다(2.4 참고).

### 2.2 스테레오 IR 쌍에서 키포인트 삼각측량

**생각.** D457 의 왼쪽, 오른쪽 IR 영상 각각에 MediaPipe(또는 다른 2D 검출기)를 돌리고, 같은 키포인트의 시차로 깊이를 구합니다. 조밀 깊이 영상의 구멍과 가장자리 문제를 피할 수 있습니다.

| 접근 | 선행 연구 | 보고된 효과 | 근거 |
|---|---|---|---|
| 다시점 MediaPipe + 삼각측량 | ThermoHands(SenSys 2025)는 1인칭과 3인칭 RGB 영상에 MediaPipe 를 돌리고(검출 임계값 0.5 → 0.1) `cv2.triangulatePoints` 로 3D 를 만든 뒤 MANO 를 맞춥니다. 카메라는 RealSense L515(머리)와 D455(외부)입니다 | 손으로 단 600여 프레임 대비 평균 관절 오차 1.01 cm(±0.34) | [확인] HTML |
| 8대 RealSense + MediaPipe | HO-Cap 은 유효한 시점 쌍마다 삼각측량 후보를 만들고 재투영 오차가 가장 작은 후보를 고르며, RANSAC 으로 거릅니다. RealSense D455 8대입니다 | 2D 재투영 3.82~4.58 px. mm 단위 정확도는 보고하지 않음 | [확인] HTML |
| 다시점 부트스트래핑 | Simon 외(CVPR 2017)는 여러 시점의 잡음 섞인 검출을 삼각측량하고 맞지 않는 것은 이상치로 버려 학습 라벨을 늘립니다 | 단일 영상 검출기가 깊이 센서 방법과 비슷한 정확도 | [확인] 초록 |
| RealSense IR 영상에 MediaPipe | EgoEV-HandPose(2026)의 라벨 작업은 RealSense D435 의 IR 영상 세 가지 변형(원본, 마스크, 배경 제거)에 MediaPipe 를 돌리고, 사람이 고른 뒤 고쳤습니다. 2D 점은 깊이와 내부 파라미터로 3D 로 올리고, 빠진 깊이는 주변 평균으로 채웠습니다 | 성공률은 보고하지 않음. 사람 수정이 필요했다는 점이 단서 | [확인] HTML 3장 |
| 수동 스테레오 손 추적 | Zhang 외(ICIP 2018)는 18,000 쌍 스테레오 벤치마크에서 스테레오 기반 추적이 능동 깊이 센서와 비슷하다고 보고합니다 | 정성 비교 | [검색] |
| 스테레오에서 바로 자세 | THU-Bi-Hand 의 방법(2019)은 스테레오 영상에서 손 자세를 회귀합니다 | 자기 데이터셋 최고 성능(수치 미확인) | [확인] 초록 |
| 상용 스테레오 IR | Meta 의 MEgATrack(SIGGRAPH 2020)은 흑백 어안 카메라 4대로 손을 추적하며 낮은 떨림을 강조합니다. Quest 의 기본 손 추적입니다 | PC 60 Hz, 모바일 30 Hz | [검색] (Meta 페이지 요약) |
| 프로젝터 켜고 끄기 | librealsense 의 `RS2_OPTION_EMITTER_ON_OFF` 는 프레임마다 프로젝터를 번갈아 켭니다. 프레임별 상태가 메타데이터로 나오는지는 확실하지 않다는 이슈가 있습니다 | 깨끗한 IR 은 절반 프레임 | [검색] |

**정리 [의견].**

- 다시점 MediaPipe 삼각측량은 데이터셋 라벨 작업에서 흔합니다(ThermoHands, HO-Cap). 시점 사이가 넓은 별도 카메라들입니다.
- **한 장치 안의 IR 스테레오 쌍에 MediaPipe 를 두 번 돌려 실시간 텔레옵의 손목 깊이를 얻는 사례는 검색 범위에서 찾지 못했습니다**(W10, W11, W12, W15).
- 1장의 계산으로 보면, 키포인트 시차의 잡음이 정확도를 정합니다. 두 영상은 거의 같고 같은 네트워크를 쓰므로 키포인트 오차의 일부는 두 영상에서 같은 방향으로 생겨 시차에서 상쇄될 수 있습니다. 하지만 MediaPipe 는 영상마다 이전 프레임으로 잘라낼 영역을 정하는 추적기라, 두 추적기가 서로 다른 영역을 잡으면 상쇄가 깨집니다. 얼마나 상쇄되는지는 재 봐야 압니다.
- 손바닥 다섯 점의 시차를 함께 쓰면(강체 맞춤이나 평균) 독립 잡음이라면 약 √5 배 줄어듭니다.
- 프로젝터를 끄면 IR 이 깨끗해지지만 조밀 깊이의 질이 떨어집니다. 번갈아 켜기를 쓰면 키포인트용 IR 과 깊이가 반씩 나옵니다.
- D400 계열의 Y8 IR 스트림이 정류된(rectified) 영상이라 시차가 같은 행에 놓인다고 알고 있으나, 이번에 원문으로 확인하지 못했습니다. 확인되면 삼각측량은 `z = f·B / (x_L - x_R)` 한 줄입니다.

### 2.3 광선 방향으로 늘어난 관측 잡음과 불확실성 필터

**생각.** 단안 PnP 의 위치 오차는 카메라에서 손목으로 가는 광선 방향으로 길쭉합니다(M2 의 깊이 대 화면 방향 2~8배). 칼만 필터의 관측 공분산을 `R = σ_l² (I - u uᵀ) + σ_r² u uᵀ`(u 는 광선 단위 벡터)로 두거나, One Euro 를 광선 성분과 옆 성분으로 나눠 따로 설정합니다.

| 접근 | 선행 연구 | 보고된 효과 | 근거 |
|---|---|---|---|
| 텔레옵에서 깊이 축 잡음을 크게 | Kwok, Koenig, Hu(2026, AKC)는 RealSense D435(480p, 10 Hz)와 MediaPipe Holistic 으로 팔을 추적합니다. 상수 속도 칼만 필터에 `R = diag(σx², σy², σz²)`, `σz ≫ σx, σy` 를 두고, 깊이가 없으면 x, y 만 갱신합니다. 가시성 0.7 미만 점은 버리고, 몸쪽 관절이 손목의 화면 폭 5% 안에 들어오면 그 깊이를 버립니다 | Vicon 대비 정지 어깨 z RMSE 34.86 → 3.27 cm(AKC 보정). 칼만 필터만의 효과는 따로 보고하지 않음. 손목 RMSE 없음 | [확인] HTML |
| 거리 제곱으로 커지는 축 방향 잡음 | Nguyen, Izadi, Lovell(2012)은 Kinect 의 축 방향과 옆 방향 잡음을 거리와 각도의 함수로 재고, 축 방향 잡음이 거리 제곱으로 커진다고 보고합니다 | 깊이 센서 잡음 모델 | [검색] |
| 비등방 점 융합 | Lefloch, Weyrich, Kolb(FUSION 2015)는 거리 영상의 점마다 비등방 잡음을 저장하고 융합에 씁니다 | 재구성 품질 개선(정성) | [검색] |
| 스테레오 삼각측량 오차 | Matthies, Shafer(1987)는 스테레오 삼각측량 오차를 3D 가우시안으로 모델링해 스칼라 모델보다 위치 추정 분산을 줄였습니다 | 시뮬레이션 | [검색] |
| 불확실성 PnP | MLPnP(Urban 외 2016)는 관측 잡음을 고려한 최대우도 PnP 와 자세 공분산을 냅니다. Vakhitov 외(CVPR 2021)는 04 문서 2.3 | 카메라 자세 추정 기준 | MLPnP [검색], Vakhitov [확인] (04 문서) |
| 단안 3D 추적의 공분산 | Brazil 외(ECCV 2020)의 단안 3D 물체 검출 추적은 신뢰도로 크기만 바꾸는 등방 공분산을 쓴다는 요약이 있습니다 | | 논문 [확인] 초록, 공분산 형태 [검색] |
| 깊이 카메라 사람 추적 필터 | Martini 외(2024)는 단일 RGB-D 카메라에서 사람 자세 추정과 깊이 카메라의 오차를 다루는 필터를 제안합니다. 초록에는 잡음 모델과 수치가 없습니다 | 정성 | [확인] 초록 |
| 신뢰도 가중 One Euro, 시점별 오차 | 04 문서 2.3 (Adzhiev 외, OAT) | | [확인] (04 문서) |

**정리 [의견].**

- 깊이 축 잡음을 옆 축보다 크게 두는 것은 RGB-D 사람 추적 텔레옵에서 이미 쓰입니다(AKC). 스테레오와 거리 센서 쪽에서는 광선 방향으로 늘어난 공분산이 오래된 표준입니다. 이 후보의 새로움은 작습니다.
- "광선 방향"과 "카메라 z 축"의 차이는 화면 가장자리에서만 커집니다. 시야각 86°의 가장자리에서는 둘 사이가 40° 넘게 벌어지므로, 손이 넓게 움직이면 광선 방향이 맞습니다. 단안 PnP 의 배율 오차는 정의상 원점에서 손목으로 가는 방향이므로(04 문서 2.1) 광선 방향 분해가 정확한 모델입니다.
- 지금 One Euro 는 성분별 속도로 차단 주파수를 올립니다. 손이 가만히 있어도 깊이 성분의 잡음이 속도로 읽히면 깊이 성분의 필터가 열립니다. 광선 성분의 차단 주파수를 옆 성분 속도(믿을 만한 쪽)로 올리고, 광선 성분 자신의 속도에는 작은 가중만 주는 변형이 간단합니다. 손을 앞뒤로만 움직일 때 지연이 생기는 비용이 있습니다.
- 이 필터는 2.4 의 깊이 융합과 같은 틀입니다. PnP 관측은 큰 `σ_r`, 센서 깊이 관측은 작은 `σ_r`, 깊이가 없으면 옆 방향만 갱신하면 됩니다.
- MediaPipe + PnP 손목 텔레옵에서 광선 방향 공분산을 쓴 사례는 검색 범위에서 찾지 못했습니다(W21, W22, W24, W26, W27, W28). 가까운 사례(AKC)가 카메라 z 축 기준으로 이미 하고 있으므로, 새로움이라기보다 적용입니다.

### 2.4 RGB-D 깊이와 RGB 키포인트의 융합 (기존 텔레옵 시스템)

| 시스템 | 손 추적 입력 | 깊이를 손에서 어떻게 읽나 | 구멍, 가장자리 처리 | 보고된 정확도 | 근거 |
|---|---|---|---|---|---|
| AnyTeleop (RSS 2023) | MediaPipe | 검출한 키포인트 화소의 깊이를 읽어 카메라 좌표 3D 점으로 만든 뒤, 손목 좌표의 키포인트와 PnP 로 맞춥니다. 여러 카메라는 처음 50 프레임에 손을 표식 삼아 회전을 맞추고, 형상 파라미터가 기준에서 덜 벗어난 카메라를 고릅니다 | 적혀 있지 않음 | 피아노 과제 틀린 건반 RGB 28.1%, RGB-D 21.8%, RGB-D 두 대 12.5%. 손목 오차 자체는 없음 | [확인] HTML |
| Vuong 외 (2021) | MediaPipe | 손목의 카메라 좌표 3D 위치를 깊이 영상에서 역투영합니다 | 적혀 있지 않음 | RGB 만으로는 집어 옮기기 33회 중 11회 성공. RGB-D 성공 수는 같은 형태로 보고되지 않음 | [확인] HTML |
| DexMV (ECCV 2022) | RGB 손 모델 + MANO 최적화 | 손 마스크(피부 분할 + 손 검출) 안 깊이 영상의 "중심"으로 루트 깊이를 정하고, 최적화에 렌더링 깊이와 관측 깊이의 마스크 차이 항(λ = 0.001)을 넣습니다. RealSense D435 두 대 | 마스크로 배경 제외 | DexYCB MPJPE 41.7(1대) → 32.5(2대 + 후처리). 단위 표기 없음 | [확인] HTML |
| DexPilot (ICRA 2020) | 깊이 점군 + PointNet++ + DART | RealSense D415 네 대 점군을 합치고, 탁자 평면을 지운 뒤 손 점만 남겨 맞춥니다 | 분할로 팔과 배경 제외. DART 가 몇 분마다 국소 최솟값에 빠진다고 씀 | 키포인트 평균 9.7 mm(검증 약 7,000장), 지연 약 1 s | [확인] ar5iv |
| GMH-D (Amprimo 외 2023) | MediaPipe | 손목 깊이 `d_wrist` 는 센서에서, 다른 관절은 `d_j = d_wrist + z_j·d_wrist` 로 MediaPipe 상대 깊이를 키웁니다. 손목을 고른 이유로 가장 안정적이고 손가락보다 넓은 면에서 추적된다는 점을 듭니다. Azure Kinect | 손목을 써서 가장자리 간섭을 피한다고 씀 | OptiTrack 대비 손가락 거리 PRMSE 8~15%(GMH 는 20% 이상). 손목 절대 위치 오차는 없음 | [확인] PDF |
| Sensors 2026 (ZED2) | MediaPipe | 정렬한 ZED2 깊이에서 랜드마크 주변 국소 중앙값을 읽고 역투영한 뒤, 지수 이동 평균과 뼈 길이 일관성 제약을 겁니다 | 유효하지 않은 화소, 구멍, 가장자리 잡음, 가림에 약하다는 점을 동기로 듭니다 | 초록에 수치 없음 | [검색] (MDPI 페이지 403) |
| Hand Shadowing (IEEE Access 2026) | MediaPipe | RealSense D435i, 640x480 30 fps. 화소 하나 `D[u, v]` 를 읽습니다 | 0.1~5.0 m 밖은 무효, 21점 중 유효 깊이가 50% 미만이면 손 전체를 버림 | 원시 끝점 프레임 간 이동 5.75 → 4.52 mm(평활 후), 위치 떨림 4.5 mm | [확인] HTML |
| AKC (2026) | MediaPipe Holistic | RealSense D435 깊이를 역투영 | 가시성과 근접 규칙으로 가린 관절 깊이 버림(2.3) | 2.3 표 | [확인] HTML |
| Robotic Telekinesis (RSS 2022) | 단일 RGB, FrankMocap | 깊이를 쓰지 않고 손목을 몸통 기준 상대 변환으로 씁니다 | 해당 없음. 이상치 거부, 저역 통과, 지수 이동 평균 | 손목 정확도 없음 | [확인] ar5iv |
| Open-TeleVision (CoRL 2024) | Vision Pro 내장 추적 | 해당 없음. 손목 자세를 SE(3) 필터로 평활합니다. VR 장치의 손 추적이 자체 시각 추적보다 대체로 안정적이라고 씁니다 | 해당 없음 | | [확인] HTML |
| Bunny-VisionPro (2024) | Vision Pro 내장 추적 | 해당 없음 | 손가락끼리 가리면 추적이 부정확해 명령이 튄다고 씀 | | [검색] |

**정리 [의견].**

- RGB 키포인트 + 깊이 영상으로 손목 위치를 구하는 것은 텔레옵에서 표준입니다. 그런데 **손 위의 어느 화소를 어떻게 읽는지는 대부분 적혀 있지 않습니다**(AnyTeleop, Vuong, AKC). 적힌 곳은 화소 하나(Hand Shadowing), 손목 하나(GMH-D), 마스크 중심(DexMV), 국소 중앙값(Sensors 2026) 정도입니다.
- **손목 텔레옵에서 깊이 샘플링 방식별 흔들림을 비교한 보고는 검색 범위에서 찾지 못했습니다**(W5, W22, W26, W30, W34). 비교 실험은 작지만 쓸모 있는 기여입니다.
- 우리에게 맞는 샘플링 방식은 다음과 같습니다. 손목 점 하나는 손목이 손등 가장자리와 팔뚝 사이에 있어 배경과 섞이기 쉽습니다. 손바닥 다섯 점으로 만든 다각형을 조금 줄여(침식) 그 안의 유효 깊이 중앙값을 손바닥 중심 깊이로 쓰고, 손목은 PnP 가 준 손바닥 중심과 손목의 상대 위치로 옮깁니다. 다각형 안 유효 화소 비율이 낮거나, 중앙값이 PnP 깊이와 크게 다르면(예: ±15%) 그 프레임은 옆 방향만 갱신합니다.
- USB 2 대역폭 안에서 컬러 + 깊이가 11 fps 였으므로, 왼쪽 IR(Y8) + 깊이를 먼저 시험할 만합니다. 깊이는 왼쪽 IR 좌표에서 나오므로 정렬 비용이 없고, 컬러와 깊이 사이의 시간 차도 없습니다 **[의견]**. 대신 MediaPipe 가 흑백, 점무늬 영상에서 손을 얼마나 잡는지가 미지수입니다(EgoEV-HandPose 는 사람 수정이 필요했습니다).

---

## 3. 새로운 시도인가

| 후보 | 선행 연구 상태 | 이 프로젝트에서의 위치 |
|---|---|---|
| 1 깊이 선생 + 작업자별 회귀기 | 깊이로 학습하고 RGB 로 추론(Cai 2018, Yuan 2018, Zhao 2020), 라벨 없는 시험 시 적응(V-HPOT) | MediaPipe 특징 위의 작업자별 작은 회귀기는 검색 범위에서 찾지 못함(W1, W4~W6, W8, W9). 작은 새로움 |
| 2 IR 스테레오 키포인트 삼각측량 | 다시점 MediaPipe 삼각측량(ThermoHands, HO-Cap), RealSense IR 에 MediaPipe(EgoEV 라벨 작업) | 한 장치의 IR 쌍으로 실시간 손목 깊이를 얻는 텔레옵은 검색 범위에서 찾지 못함(W10~W12, W15). 작은 새로움 |
| 3 광선 방향 필터 | 깊이 축 잡음을 크게 둔 칼만 필터(AKC), 비등방 깊이 잡음 모델(Nguyen 2012, Lefloch 2015) | 광선 방향 분해 자체는 알려진 방법의 적용 |
| 4 RGB-D 융합 | AnyTeleop, Vuong, DexMV, GMH-D, Sensors 2026, Hand Shadowing, AKC | 알려진 방법. 샘플링 방식별 흔들림 비교는 검색 범위에서 찾지 못함 |
| 4' 왼쪽 IR + 깊이로 USB 2 우회 | 개별 요소는 알려짐 | 텔레옵에서 이 조합을 쓴 보고는 검색 범위에서 찾지 못함(W15, W19). 공학적 선택 |

---

## 4. 순위 (우리 설정 기준)

기대 효과는 R4 의 깊이 방향 흔들림(p95 42~43 mm)에 대한 것이고, 모두 **[의견]** 입니다.

| 순위 | 후보 | 기대 효과 | 노력 | 새로움 | 무엇을 잴까 |
|---|---|---|---|---|---|
| 1 | 4 + 4' 센서 깊이 융합. 먼저 왼쪽 IR(Y8) + 깊이를 USB 2 에서, 안 되면 USB 3 케이블로 컬러 + 깊이 | 높음. 평면 기준 센서 잡음은 1 mm 안팎이므로, 샘플링만 잘하면 깊이 방향 흔들림이 수 mm 로 내려갈 수 있습니다. 가장자리와 구멍에서 튀는 프레임이 위험입니다 | 낮음~중간. 스트림 설정, 다각형 중앙값, 문턱 하나 | 낮음 | USB 2 에서 IR + 깊이 조합별 실제 fps, IR 영상에서 MediaPipe 검출률(컬러 대비), 정지 손의 깊이 방향 프레임 간 이동 p50/p95, 유효 깊이 비율, 센서 깊이와 PnP 깊이의 차이 분포, 손등과 손바닥 각각 |
| 2 | 3 광선 방향 필터 | 중간. 지연과 맞바꾸는 개선이며 원인을 없애지는 않습니다. 1순위와 같은 틀이라 버릴 코드가 아닙니다 | 매우 낮음. 녹화 자료로 오프라인 비교 가능 | 낮음 | 같은 녹화에서 지금 One Euro 와 비교: 깊이 방향과 옆 방향 p95, 손을 앞뒤로 움직일 때의 지연(상호상관 최대 시점), 손을 옆으로 움직일 때 지연이 늘지 않는지 |
| 3 | 2 IR 스테레오 키포인트 삼각측량 | 낮음~중간. 시차 잡음이 0.5 px 아래여야 지금보다 뚜렷이 낫습니다(1장 계산). 깊이 구멍을 피하는 보조 경로로는 쓸모가 있습니다 | 중간. 두 추적기 운영, 프로젝터 설정, 정류 확인 | 작음 | 정지 손에서 손바닥 다섯 점 시차의 표준편차(px), 이를 깊이로 바꾼 값과 조밀 깊이 중앙값의 비교, 두 영상의 검출 동시 성공률, 프로젝터 켬, 끔, 번갈아 각각 |
| 4 | 1 깊이 선생 + 작업자별 회귀기 | 고정 편향에는 높음, 흔들림에는 불확실. 깊이가 없는 카메라(노트북 웹캠)로 돌아갈 때 가치가 있습니다 | 중간. 1순위의 기록을 그대로 라벨로 씀 | 작은 새로움 | 다른 세션(같은 작업자)에서의 깊이 오차 평균과 p95, 손바닥과 손등 전환 때의 깊이 계단, 프레임 간 흔들림이 PnP 보다 줄었는지, 입력 특징별 기여(배율, 이미지 크기, 법선 각도) |

**추천 순서.**

1. **오늘(로봇 없이, 카메라만).** USB 2 에서 왼쪽 IR + 깊이 스트림이 어떤 해상도와 fps 로 함께 열리는지 확인합니다. 동시에 R4 같은 녹화에 광선 방향 One Euro 변형을 오프라인으로 돌려 봅니다.
2. **IR + 깊이가 열리면** 손을 고정대에 올리고 손바닥과 손등으로 각각 10 s 씩 녹화합니다. PnP 깊이, 손목 점 하나의 센서 깊이, 손바닥 다각형 중앙값을 같은 프레임에서 나란히 기록합니다. 이 녹화가 1순위의 효과, 2순위 필터의 `σ_r` 설정, 4순위의 학습 라벨을 한꺼번에 줍니다.
3. **IR 에서 MediaPipe 가 잘 안 되면** USB 3 케이블로 컬러 + 깊이를 정렬해 같은 측정을 합니다.
4. 3순위와 4순위는 2단계의 녹화로 오프라인에서 먼저 판단합니다. 3순위는 오른쪽 IR 도 함께 녹화해야 합니다.

실물 팔을 움직이는 시험은 위 측정 뒤에, 총괄 에이전트가 사용자 승인을 받아 합니다.

---

## 5. 확인하지 못한 것

- Intel 의 깊이 RMS 식과 subpixel 0.05~0.1 은 포럼에 올라온 튜닝 자료의 검색 요약으로만 봤습니다. Intel 자가 보정 문서는 같은 변수를 정의하지만 식은 그림이라 읽지 못했습니다.
- D457 의 "USB 단자는 생산과 디버그용"이라는 문구는 Intel 보도자료의 검색 요약입니다. 페이지가 열리지 않았습니다. 사실이라면 USB 3 케이블로도 공식 지원 모드가 제한될 수 있습니다.
- USB 2 에서 열거되는 IR, 깊이 모드는 사용자 보고(D400 계열)이고 D457 에서 확인하지 않았습니다. Intel 의 USB 2 지원 페이지는 이번에 열리지 않았습니다.
- D400 의 Y8 IR 스트림이 정류된 영상인지는 원문으로 확인하지 못했습니다.
- Sensors 2026(ZED2) 논문은 MDPI 페이지가 403 이라 초록 요약만 봤고 수치가 없습니다.
- Remelli 외(ICCV 2017), MonoLift, Lienen 외(ACML 2021), Zhang 외(ICIP 2018), MEgATrack 의 세부, Nguyen 외(2012), Lefloch 외(2015), Matthies, Shafer(1987), MLPnP, Bunny-VisionPro 의 "가림 때 튀는 명령"은 검색 요약만 봤습니다.
- Brazil 외(ECCV 2020)의 등방 공분산 서술은 검색 요약이고, 초록에는 칼만 필터 언급이 없었습니다.
- AnyTeleop, Vuong 외, AKC 는 손 위 깊이 샘플링 방식을 적지 않았습니다. 코드는 보지 않았습니다.
- GMH-D 의 오차는 손가락 사이 거리 기준이라 손목 절대 위치의 정확도는 말해 주지 않습니다.
- 1장의 시차 계산은 HFOV 87°, 거리 0.6 m 가정입니다. 에고 카메라에서 손까지의 실제 거리는 재지 않았습니다.

---

## 참고문헌

| 문헌 | 링크 | 확인 수준 |
|---|---|---|
| Cai, Ge, Cai, Yuan, Weakly-supervised 3D Hand Pose Estimation from Monocular RGB Images, ECCV 2018 | [mlanthology](https://mlanthology.org/eccv/2018/cai2018eccv-weaklysupervised) | [확인] 초록 |
| Yuan, Stenger, Kim, RGB-based 3D Hand Pose Estimation via Privileged Learning with Depth Images, 2018 | [arXiv:1811.07376](https://arxiv.org/abs/1811.07376) | [확인] 초록 |
| Zhao 외, Knowledge as Priors: Cross-Modal Knowledge Generalization, CVPR 2020 | [arXiv:2004.00176](https://arxiv.org/abs/2004.00176) | [확인] 초록 |
| Mucha, Wray, Kampel, Towards Egocentric 3D Hand Pose Estimation in Unseen Domains (V-HPOT), WACV 2026 | [arXiv:2601.06537](https://arxiv.org/abs/2601.06537) | [확인] 초록 |
| Remelli 외, Low-Dimensionality Calibration Through Local Anisotropic Scaling for Robust Hand Model Personalization, ICCV 2017 | [CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Remelli_Low-Dimensionality_Calibration_Through_ICCV_2017_paper.html) | [검색] (403) |
| Wang 외, MonoLift, NeurIPS 2025 | [NeurIPS](https://neurips.cc/virtual/2025/poster/115316) | [검색] |
| Lienen 외, Robust Regression for Monocular Depth Estimation, ACML 2021 | [PMLR](https://proceedings.mlr.press/v157/lienen21a.html) | [검색] |
| Ding 외, ThermoHands, SenSys 2025 | [arXiv:2403.09871](https://arxiv.org/abs/2403.09871) | [확인] HTML 라벨 절 |
| HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction, 2024 | [arXiv:2406.06843](https://arxiv.org/abs/2406.06843) | [확인] HTML |
| Simon 외, Hand Keypoint Detection in Single Images using Multiview Bootstrapping, CVPR 2017 | [arXiv:1704.07809](https://arxiv.org/abs/1704.07809) | [확인] 초록 |
| Wang 외, EgoEV-HandPose, 2026 | [arXiv:2605.12297](https://arxiv.org/abs/2605.12297) | [확인] HTML 3장 |
| Zhang 외, 3D Hand Pose Tracking and Estimation Using Stereo Matching (ICIP 2018 벤치마크) | [Birmingham](https://research.birmingham.ac.uk/en/publications/a-hand-pose-tracking-benchmark-from-stereo-matching/) | [검색] |
| End-to-End 3D Hand Pose Estimation from Stereo Cameras (THU-Bi-Hand), 2019 | [arXiv:1902.09795](https://arxiv.org/abs/1902.09795) | [확인] 초록 |
| Han 외, MEgATrack, SIGGRAPH 2020 | [Meta AI](https://ai.meta.com/research/publications/megatrack-monochrome-egocentric-articulated-hand-tracking-for-virtual-reality/) | [검색] |
| librealsense 이슈, 프로젝터 번갈아 켜기와 프레임 상태 | [GitHub #1299](https://github.com/IntelRealSense/librealsense/issues/1299) | [검색] |
| Kwok, Koenig, Hu, Seeing Through Occlusion: Deterministic Arm Kinematic Correction for Robot Teleoperation, 2026 | [arXiv:2606.19240](https://arxiv.org/abs/2606.19240) | [확인] HTML |
| Nguyen, Izadi, Lovell, Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking, 3DIMPVT 2012 | [CSDL](https://computer.org/csdl/proceedings-article/3dimpvt/2012/4873a524/12OmNAlvHUP) | [검색] |
| Lefloch, Weyrich, Kolb, Anisotropic Point-Based Fusion, FUSION 2015 | [PDF](https://reality.tf.fau.de/projects/kinect/lefloch15anisotropic.pdf) | [검색] (PDF 용량 초과로 못 엶) |
| Matthies, Shafer, Error Modeling in Stereo Navigation, IEEE J. Robot. Autom. 1987 | [CMU RI](https://www.ri.cmu.edu/publications/error-modeling-in-stereo-navigation) | [검색] |
| Urban, Leitloff, Hinz, MLPnP, 2016 | [arXiv:1607.08112](https://arxiv.org/abs/1607.08112) | [검색] |
| Brazil 외, Kinematic 3D Object Detection in Monocular Video, ECCV 2020 | [arXiv:2007.09548](https://arxiv.org/abs/2007.09548) | [확인] 초록, 공분산 형태는 [검색] |
| Martini 외, A Robust Filter for Marker-less Multi-person Tracking in HRI Scenarios, 2024 | [arXiv:2406.01832](https://arxiv.org/abs/2406.01832) | [확인] 초록 |
| Qin 외, AnyTeleop, RSS 2023 | [arXiv:2307.04577](https://arxiv.org/abs/2307.04577) | [확인] HTML 4장, 표 IV |
| Vuong 외, Single RGB-D Camera Teleoperation for General Robotic Manipulation, 2021 | [arXiv:2106.14396](https://arxiv.org/abs/2106.14396) | [확인] HTML |
| Qin 외, DexMV, ECCV 2022 | [arXiv:2108.05877](https://arxiv.org/abs/2108.05877) | [확인] HTML |
| Handa 외, DexPilot, ICRA 2020 | [arXiv:1910.03135](https://arxiv.org/abs/1910.03135) | [확인] ar5iv |
| Amprimo 외, Hand tracking for clinical applications: GMH and GMH-D, 2023 | [arXiv:2308.01088](https://arxiv.org/abs/2308.01088) | [확인] PDF 본문 |
| A Low-Cost RGB-D Sensing Front-End for Stable 3D Hand Landmark Reconstruction Using MediaPipe and ZED2 Stereo Depth, Sensors 26(12) 3730, 2026 | [MDPI](https://www.mdpi.com/1424-8220/26/12/3730) | [검색] (403) |
| Chiche 외, Vision-Based Hand Shadowing for Robotic Manipulation via Inverse Kinematics, IEEE Access 2026 | [arXiv:2603.11383](https://arxiv.org/abs/2603.11383) | [확인] HTML |
| Sivakumar, Shaw, Pathak, Robotic Telekinesis, RSS 2022 | [arXiv:2202.10448](https://arxiv.org/abs/2202.10448) | [확인] ar5iv |
| Cheng 외, Open-TeleVision, CoRL 2024 | [arXiv:2407.01512](https://arxiv.org/abs/2407.01512) | [확인] HTML |
| Ding 외, Bunny-VisionPro, 2024 | [arXiv:2407.03162](https://arxiv.org/abs/2407.03162) | [검색] |
| FRAMOS, Depth Camera D457 제품 페이지 | [FRAMOS](https://framos.com/en/products/depth-camera-d457-camera-only-26703) | [확인] |
| Rutronik, D457 Product Brief | [PDF](https://www.rutronik.com/fileadmin/Rutronik/News/Knowledge/Product_News/2023/KW05/D457_Product_Brief_Brochure.pdf) | [검색] |
| RealSense, Self-Calibration for D400 Series (Subpixel RMS 정의) | [dev.realsenseai.com](https://dev.realsenseai.com/?p=41425) | [확인] 변수 정의, 식은 그림이라 못 읽음 |
| Intel, BKMs for Tuning RealSense D4xx Cameras (RMS 식, MinZ) | [포럼 첨부](https://community.intel.com/cipcp26785/attachments/cipcp26785/feeditem_no_topic/6364/1/BKMs_Tuning_RealSense_D4xx_Cam.pdf) | [검색] (404) |
| RealSense 지원 포럼, USB 2 에서 D400 모드 목록 | [포럼](https://support.realsenseai.com/hc/en-us/community/posts/1500001389542/comments/1500001178482) | [검색] |

04 문서에서 가져온 문헌(OAT, Dexterous Point Policy, Kong 외, Snap 특허, Vakhitov 외, Adzhiev 외)은 04 문서의 참고문헌과 확인 수준을 따릅니다.

이 저장소의 근거: `docs/05_ego_camera.md`(4장 USB), `docs/results/r4_ego_right_wrist_follow.md`(손목 흔들림 표), `docs/results/m2_scale_jitter.md`(3장, 7장), `vision/filters.py`, `vision/hand_tracker.py`(`POSITION_FILTER`), `vision/wrist_pose.py`(SQPnP, 배율 비례).

---

## 부록 A. 검색어

2026-10-19 에 웹 검색 도구로 실행했습니다. 04 문서의 Q1~Q39 와 구별하려고 W 로 번호를 붙였습니다.

| 번호 | 검색어 | 주제 |
|---|---|---|
| W1 | RGB-D depth as supervision to correct monocular hand depth estimation per-user calibration regressor | 1 |
| W2 | self-supervised 3D hand pose estimation depth camera supervision RGB test-time adaptation absolute root depth | 1 |
| W3 | test-time adaptation hand pose estimation SeqHAND OR "test-time adaptation" 3D hand mesh | 1 |
| W4 | learning to estimate distance from apparent hand size monocular camera regression calibration depth sensor ground truth | 1 |
| W5 | MediaPipe hand landmarks RealSense depth calibration regression wrist distance estimation RGB only after training | 1, 4 |
| W6 | online self-calibration hand scale monocular tracking using depth sensor then RGB-only camera student teacher | 1 |
| W7 | Cai Ge Cai Yuan "Weakly-supervised 3D Hand Pose Estimation from Monocular RGB Images" arXiv depth regularizer | 1 |
| W8 | personalized depth correction hand tracking learned from RGB-D sensor deploy RGB-only webcam user-specific calibration distance estimation hand | 1 |
| W9 | RGB-D teacher distillation monocular hand pose metric depth student RGB only camera-space wrist translation knowledge distillation | 1 |
| W10 | RealSense infrared stereo pair hand keypoint triangulation MediaPipe on left and right IR images emitter off | 2 |
| W11 | stereo hand pose estimation triangulation 2D keypoints binocular camera 3D hand tracking accuracy compared depth sensor | 2 |
| W12 | multi-view camera hand keypoint triangulation teleoperation dexterous hand DexCap OR "multi-view" MediaPipe triangulation robot teleop | 2 |
| W13 | Leap Motion Ultraleap stereo infrared hand tracking accuracy depth error mm evaluation | 2 |
| W14 | MEgATrack monochrome egocentric articulated hand tracking multiple cameras triangulation jitter SIGGRAPH 2020 | 2 |
| W15 | "MediaPipe" infrared frames RealSense D435 back-projected depth hand keypoints dataset refinement | 2, 4 |
| W16 | Intel RealSense D400 depth RMS error formula distance squared subpixel focal length baseline tuning whitepaper | 장치 |
| W17 | Intel RealSense D457 datasheet baseline 95 mm global shutter IR resolution GMSL depth accuracy | 장치 |
| W18 | librealsense RS2_OPTION_EMITTER_ON_OFF alternate frames emitter on off D400 frame metadata laser power | 2 |
| W19 | RealSense D400 USB 2.0 supported stream profiles infrared Y8 depth simultaneous limited bandwidth | 장치, 4 |
| W20 | RealSense D400 minimum depth distance MinZ decreases lower resolution 848x480 480x270 hand close range | 장치 |
| W21 | anisotropic measurement covariance along camera ray Kalman filter depth uncertainty monocular 3D tracking | 3 |
| W22 | Kalman filter hand tracking teleoperation depth axis noise larger than lateral covariance RGB-D hand position | 3, 4 |
| W23 | Kinect depth noise model axial lateral noise standard deviation distance squared Nguyen Izadi Lovell 2012 | 3 |
| W24 | uncertainty-aware 3D human pose filtering heteroscedastic covariance monocular depth direction Kalman smoothing keypoints | 3 |
| W25 | Matthies Shafer 1987 "Error modeling in stereo navigation" 3D Gaussian covariance elongated along line of sight | 3 |
| W26 | MediaPipe hand landmarks Kalman filter 3D wrist teleoperation measurement noise covariance tuning per axis depth | 3, 4 |
| W27 | monocular PnP pose uncertainty translation covariance largest along optical axis depth direction ellipsoid | 3 |
| W28 | "line-of-sight" OR "viewing ray" anisotropic noise model filtering 3D keypoints depth camera human tracking robot | 3 |
| W29 | Lefloch Weyrich Kolb anisotropic point-based fusion axial lateral noise covariance 2015 | 3 |
| W30 | DexMV hand pose estimation RGB-D depth MediaPipe 3D hand joints optimization RealSense camera video | 4 |
| W31 | DexPilot depth-based hand tracking four RealSense cameras point cloud PointNet hand pose teleoperation accuracy | 4 |
| W32 | Robotic Telekinesis wrist 3D position single RGB camera FrankMocap camera translation smoothing Sivakumar Pathak | 4 |
| W33 | Bunny-VisionPro OpenTeleVision wrist tracking Apple Vision Pro low-pass filter hand pose teleoperation dexterous | 4 |
| W34 | Sensors 2026 depth-enhanced MediaPipe ZED2 stereo depth local median landmark back-projection bone-length constraint hand | 4 |

검색 도구는 미국 지역 결과만 돌려줍니다. 학회 논문집 내부 검색(IEEE Xplore, ACM DL)은 하지 않았고, MDPI, CVF, Intel 일부 페이지는 403, 404, DNS 오류로 열리지 않았습니다.
