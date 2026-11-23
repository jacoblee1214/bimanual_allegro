# 깊이 공백 보정 조사 v2: v1 인용 검증, v4 한계, USB 2.1

- 날짜: 2026-11-23
- 작성 에이전트와 모델: Kiro 서브에이전트(선행연구 검증 담당), claude-opus-5.5
- 입력 문서: `AGENTS.md`, `docs/09_gap_hold_survey.md`(v1, 검증 대상), `docs/06_wrist_depth_survey.md`, `docs/04_rnd_candidates_survey.md`, `docs/05_ego_camera.md`, `docs/results/m8_gap_v4.md`, `docs/results/m5_hand_scale.md`, `docs/results/m7_hold_v3.md`, `docs/results/r5_ego_bimanual_follow.md`
- 이 문서는 조사입니다. 새로 잰 값은 없습니다. 카메라, 로봇, 네트워크 장비, git, 패키지 설치는 쓰지 않았습니다. 저장소에서는 파일을 읽기만 했습니다(`m5_detections_ego_depth_424.npz` 의 키 목록 확인 포함). 웹 페이지 안의 지시문은 데이터로만 다뤘습니다.
- v1 은 고치지 않았습니다(`AGENTS.md` 버전 규칙).

**표기.**

| 표기 | 뜻 |
|---|---|
| **[열람]** | 이번에 원문(초록 페이지, HTML 본문, README, LICENSE, 공식 문서, 이슈 본문)을 직접 열어 확인했습니다 |
| **[검색]** | 검색 결과 요약만 봤습니다. 원문은 열지 못했습니다 |
| **[저장소]** | 이 저장소 문서나 파일에 있는 값입니다 |
| **[계산]** | 이 문서에서 식으로 낸 값입니다. 가정을 같이 적었습니다 |
| **[의견]** | 추측입니다. 사실이 아닙니다 |

연도는 arXiv 제출일, 페이지 날짜, 이슈 작성일입니다. 인용은 30단어를 넘기지 않고 풀어 썼습니다.

---

## 0. v1 대비 바뀐 점

**정정된 인용**

1. **HaMeR 라이선스.** v1 은 "README 에 없음, LICENSE 파일은 열지 않음"이라고 했습니다. 저장소의 `LICENSE.md` 는 **MIT**(2023, UC Regents, Georgios Pavlakos)입니다 [열람]. 다만 MIT 는 코드에만 적용되고 MANO 모델 파일은 따로 MANO 라이선스를 따릅니다. 그래서 상업 사용 가능 여부는 MANO 조건이 정합니다 [의견].
2. **Ferstl 외 2021 의 "운동량으로 궤적을 예측하는 방법은 좋지 않았다".** 초록(Semantic Scholar API, ACM SAP 2021)에는 세 가지 완화 전략이 모두 효과가 있었다는 내용만 있고, 운동량 예측 이야기는 없습니다 [열람]. 저자 PDF 는 이 도구로 열 수 없었고(PDF 형식 미지원), ACM 페이지는 403 이었습니다. **이 문장은 이번에 확인하지 못했습니다.** v1 과 04 문서는 이 문장을 근거로 "감쇠 속도 외삽"을 3순위로 낮췄는데, 그 근거는 지금 미확인입니다.
3. **MT-π(Motion Tracks).** v1 은 "2D 궤적이라 깊이 오차의 영향을 받지 않음"이라고 적었습니다. 초록에 따르면 시험 때 **두 카메라 시점**의 2D 궤적으로 6DoF 를 복원합니다 [열람]. 카메라 하나인 이 저장소에서는 깊이 정보가 따로 필요하므로, v1 의 장점 서술은 이 저장소에 맞지 않습니다.
4. **EgoVLA 의 MANO.** v1 은 "[확인, 초록]"이라 했지만 초록에는 MANO 가 없습니다. 본문 그림 2와 3.2절에 손목 자세(카메라 좌표) + MANO 상위 15개 PCA 성분이 행동 공간으로 나옵니다 [열람]. 내용은 맞고 확인 위치가 달랐습니다. v1 9장은 같은 논문을 "검색 요약만"에도 넣어 표기가 서로 어긋났습니다.
5. **EgoScale 의 위험 서술.** v1 은 "22-DoF 손 대상이라 Allegro(16-DoF)와 다름"을 위험으로 적었습니다. 초록과 페이지는 22-DoF 손으로 평가하면서 **더 적은 자유도의 손으로도 잘 옮겨졌다**고 씁니다 [열람]. 위험이라기보다 열린 질문입니다.
6. **D457 의 USB 지위.** v1 은 다루지 않았습니다. 이 저장소의 카메라는 D457(SDK 에는 D455 로 보고)입니다 [저장소: 05 문서 2장]. RealSense 포럼에서 Intel 측 답변자는 D457 이 GMSL/FAKRA 용으로 설계됐고 USB 포트는 펌웨어 갱신과 디버그용이며 "USB plug and play 카메라가 아니다"라고 답했습니다 [열람, 2022]. 06 문서에서 [검색]이던 내용이 확인됐습니다.

**확인된 것(v1 그대로).** EgoHand 의 수치, 지연, 라이선스, NVIDIA 소속. WiLoR, HaWoR 의 CC-BY-NC-ND. Dexterous Point Policy, SpatialDex, InfiniHand, FounRef, MOMA, IMM-MOT, Motion Dynamics KF, Mueller 2017 의 존재와 날짜. 1장 표에 하나씩 적었습니다.

**새로 넣은 것.** v4 한계에 대한 선행연구(2장), RealSense 깊이 공백 원인과 설정(2.3절), 복귀 계단 완화의 지연 비용(2.4절), USB 2.1 원인과 공식 USB 2 모드(3장), 순위 재조정(5장).

**순위 변화.** v1 1순위(span 비율)는 이미 v4 로 구현됐습니다(`m8_gap_v4.md`). v1 3순위(속도 외삽)는 근거가 미확인이 되어 뺐습니다. 새 순위는 5장에 있습니다.

---

## 1. 사실 확인 표 (v1 인용)

| # | v1 의 주장 | 판정 | 원문에서 본 것 | 출처(연도) |
|---|---|---|---|---|
| a1 | EgoHand 는 NVIDIA 의 모델이다 | 확인됨 | 부록 A 의 핵심 기여자 소속이 NVIDIA. 본문에 NVIDIA OSMO, SOMA-X 사용 | https://arxiv.org/html/2610.30293 (2026) [열람] |
| a2 | 2026-10 제출, v3 2026-11-18 | 확인됨 | v1 2026-10-21, v2 2026-10-26, v3 2026-11-18 | https://arxiv.org/abs/2610.30293 [열람] |
| a3 | 모델과 코드 OpenMDW, 상업 사용 허용 | 확인됨 | 초록과 본문 1장에 같은 서술 | 같은 곳 [열람] |
| a4 | H2O RTE 평균 25.6 mm(p90 42.1) | 확인됨 | 표 1, 평활 적용 25.63 / 42.10, 평활 없음 25.70 / 42.26 | 같은 곳, 표 1 [열람] |
| a5 | HOT3D RTE 36.1 mm(p90 57.6) | 확인됨 | 표 2, 평활 적용 36.13 / 57.63 | 같은 곳, 표 2 [열람] |
| a6 | 정답 손 박스를 입력으로 준 조건 | 확인됨 | 6.1절 평가 규약 | 같은 곳 [열람] |
| a7 | 지연 18.35 + 2.90 ms(L40), 9.01 ms(RTX PRO 6000) | 확인됨 | 표 5. RTX PRO 6000 은 추론 9.01 + 후처리 1.86 = 10.87 ms. 배치 32, 검출과 크롭 제외 | 같은 곳, 6.4절 [열람] |
| a8 | WiLoR, HaWoR 의 H2O RTE 값 | 찾을 수 없음 | 표 1 의 해당 행이 이번에도 잘려 나와 읽지 못함 | 같은 곳 |
| b1 | WiLoR 모델 CC-BY-NC-ND | 확인됨 | README 의 License 절. Ultralytics 와 MANO 는 별도 라이선스 | https://github.com/rolpotamias/WiLoR (README, 2026-03 갱신 표기) [열람] |
| b2 | HaWoR 모델 CC-BY-NC-ND | 확인됨 | README 의 License 절. MANO 별도 | https://github.com/ThunderVVV/HaWoR (2025) [열람] |
| b3 | WiLoR `--fast` 최대 1.6배 | 확인됨 | README | 위 WiLoR [열람] |
| c1 | Ferstl 외 2021: 세 전략(fade, rest, hybrid) 모두 효과 | 확인됨 | 초록: 세 완화 전략이 모두 영향을 줄였다 | DOI 10.1145/3474451.3476235, Semantic Scholar API (2021) [열람] |
| c2 | Ferstl 외 2021: 운동량 궤적 예측은 좋지 않았다 | 찾을 수 없음 | 초록에 없음. 본문 PDF 를 열지 못함 | 저자 PDF https://www.cs.ucdavis.edu/~neff/papers/HandTrackingLossAndStudyDesign_SAP21.pdf (열기 실패) |
| d1 | EgoVLA(2025-07): 손목 자세 + MANO 를 IK 와 재타게팅으로 로봇 행동으로 | 확인됨(위치 정정) | 초록: IK 와 재타게팅. 본문 그림 2, 3.2절: 손목 자세 + MANO PCA 15 | https://arxiv.org/abs/2507.12440 (2025-07-16) [열람] |
| d2 | EgoScale(2026-02): 20,854 시간, 22-DoF | 확인됨(위험 서술 정정) | 20,854 시간, 22-DoF, 더 적은 자유도 손으로도 전이 | https://arxiv.org/abs/2602.16710 (2026-02-18), https://research.nvidia.com/labs/gear/egoscale/ (2026-02-19) [열람] |
| d3 | Dexterous Point Policy(2026-06): 손목과 손끝 3D 키포인트, 로봇 시연 없이 | 확인됨 | 초록: 손목과 손끝 수준에서 사람과 로봇 행동이 가깝게 맞음, 75.0% 대 VLA 1.0% | https://arxiv.org/abs/2606.10614 (2026-06-09) [열람] |
| d4 | SpatialDex(2026-10): 손 크기 사전 지식으로 영상 단위 배율 정렬 | 확인됨 | 초록. 정식 제목은 "Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos" | https://arxiv.org/abs/2610.29716 (2026-10-21) [열람] |
| d5 | MT-π(2025-01): 2D 궤적이라 깊이 오차 영향 없음 | 다름 | 시험 때 두 시점의 2D 궤적으로 6DoF 복원. 4개 실제 과제 평균 86.5% | https://arxiv.org/abs/2501.06994 (2025-01-13) [열람] |
| e1 | HaMeR 코드 라이선스 미확인 | 다름: MIT | `LICENSE.md` = MIT. MANO 는 별도 | https://raw.githubusercontent.com/geopavlakos/hamer/main/LICENSE.md (2023) [열람] |
| e2 | HaMeR H2O RTE 37.25(p90 71.41), HOT3D 51.92 | 확인됨 | EgoHand 표 1, 2 | EgoHand HTML [열람] |
| e3 | HaMeR, WiLoR, HaWoR 지연 26.00 / 27.43 / 26.64 ms(L40) | 확인됨 | EgoHand 표 5 | EgoHand HTML [열람] |
| f1 | InfiniHand(2026-09) 11.19 FPS, HaWoR 의 두 배 이상, GPU 미기재 | 확인됨 | 초록 | https://arxiv.org/abs/2609.35743 (2026-09-28) [열람] |
| f2 | FounRef(2026-09) 단안 사전 + 희소 앵커 | 확인됨 | MoGe-2 와 LiDAR 앵커로 구현, 학습 없음 | https://arxiv.org/abs/2609.29224 (2026-09-24) [열람] |
| f3 | MOMA(2025-06) | 확인됨 | 보정 때 희소 정답 깊이로 한 번 정렬, IROS 2025 | https://arxiv.org/abs/2506.17110 (2025-06-20) [열람] |
| f4 | IMM-MOT(2025) 자동차 3D 추적 | 확인됨 | nuScenes AMOTA 73.8% | https://arxiv.org/abs/2502.09672 (2025-02-13) [열람] |
| f5 | Motion Dynamics KF(2025): 단일 KF 가 지연이 낮아 널리 쓰임 | 확인됨 | 초록. KITTI, Waymo | https://arxiv.org/abs/2505.07254 (2025-05, v4 2026-10) [열람] |
| f6 | Mueller 외 2017: 깊이로 손 중심 크롭, 공백 보정 없음 | 확인됨 | 초록, ICCV 2017 | https://arxiv.org/abs/1704.02201 [열람] |
| f7 | Sensors 26(12) 3730: 랜드마크 주변 중앙값, EMA, 뼈 길이 제약 | 확인됨(검색 수준) | 제목과 초록 요약: MediaPipe + ZED2, 국소 중앙값이 유효 깊이 비율을 0.9731 → 0.9738 로 조금 올림. MDPI 와 PMC 본문은 403 | https://www.mdpi.com/1424-8220/26/12/3730 (2026) [검색] |

**존재를 확인할 수 없는 논문.** v1 이 인용한 논문 중 존재 자체를 확인하지 못한 것은 없습니다. 확인하지 못한 것은 Ferstl 본문의 특정 문장(c2)과 EgoHand 표의 잘린 행(a8)입니다.

---

## 2. v4 한계에 관한 선행연구

v4 는 공백 동안 `거리 = 마지막 센서 거리 × (앵커 span / 지금 span)^0.5` 입니다. span 은 랜드마크 0, 1, 5, 9, 13, 17 의 중심으로부터의 RMS 화소 거리입니다 [저장소: `m8_gap_v4.md` 2장].

### 2.1 단안 2D 손 크기로 거리 추정할 때 회전, 원근 단축, 손가락 자세

**선행연구.**

- **Iqbal 외 2018(2.5D 히트맵, NVIDIA/Bonn).** 2D 키포인트와 "루트 기준 상대 깊이"가 있고 뼈 하나의 3D 길이 C 를 알면, 루트(손바닥)의 절대 깊이가 2차 방정식의 해로 정확히 정해진다고 보입니다(식 3~7). 기준 뼈로 손바닥과 검지 MCP 사이를 골랐고, 2D 추정이 가장 안정적인 점들이라서라고 씁니다 [열람, https://arxiv.org/abs/1804.09534 (2018), ar5iv 3.2~3.3절]. **이 방식은 2D 길이만 쓰지 않고 상대 깊이를 함께 쓰므로 회전에 따른 단축이 식 안에서 보정됩니다.** v4 의 span 비율은 이 식에서 상대 깊이를 0 으로 둔 특수한 경우입니다 [의견].
- **Google, MediaPipe 손 추적 개선(Sung 외 2021).** 원래 MediaPipe 는 손목→중지 MCP 의 2D 벡터로 회전과 배율을 정했는데, 손바닥이 카메라를 정면으로 볼 때 이 거리가 0 에 가까워져 불안정했다고 씁니다. 그래서 검지, 중지, 새끼 MCP 의 평균을 중심으로 두고, 중심에서 가장 먼 MCP 까지 거리를 배율로 쓰며, 회전은 서로 거의 수직인 두 벡터(중지 MCP→손목, 검지 MCP→새끼 MCP)의 합에서 구합니다. ASL 검증 세트에서 71.3 대 66.5 mAP 였습니다 [열람, https://arxiv.org/html/2111.00038 (2021) 2.1절]. **한 방향 길이는 회전에 약하고, 서로 수직인 두 길이를 함께 쓰면 안정적이라는 근거입니다.**
- 깊이 공백 동안 손 크기로 거리를 이은 선행연구는 이번에도 찾지 못했습니다(검색어 9장). 없다는 뜻은 아닙니다.

**오차 크기 [계산].** 핀홀 모델에서 span 선이 회전축에 수직이면 span 은 cos θ 배가 됩니다. gain 1 이면 거리 과대 추정은 θ = 15° / 30° / 45° 에서 3.5% / 15.5% / 41.4% 이고, gain 0.5 이면 1.7% / 7.5% / 18.9% 입니다. **gain 0.5 는 회전 오차를 반으로 줄이지만 실제 앞뒤 이동도 반만 따라갑니다.** `m8` 에서 gain 1.0 이 0.5 보다 모든 경우에 나빴다는 결과는 이 영상에서 span 변화의 상당 부분이 거리 변화가 아닌 것(회전, 자세, 잡음)이라는 뜻일 수 있습니다 [의견, `m8` 4.3절의 추측과 같음].

**이 저장소에 붙일 수 있는 보정 [의견, 시험 전].** 모두 `m5_detections_ego_depth_424.npz` 에 이미 있는 값만 씁니다. 이 파일에는 `*_image`(21×3, z 포함), `*_world`(21×3), `*_R`(PnP 회전)이 있습니다 [저장소, 키 확인].

| 변형 | 방법 | 비용 | 위험 |
|---|---|---|---|
| S1 두 축 최대 비율 | 손바닥 세로(손목→중지 MCP)와 가로(검지 MCP→새끼 MCP)의 span 비율을 따로 구하고, 둘 중 큰 쪽(덜 줄어든 쪽)을 거리 변화로 씀. 한 축 회전은 한 방향만 줄인다는 가정. Sung 외 2021 의 "두 수직 벡터" 생각을 옮긴 것 | 몇 줄 | 비스듬한 축 회전은 두 방향을 다 줄임. 손가락을 오므리면 가로 span 이 줄어듦 |
| S2 이미지 z 를 넣은 3D span | MediaPipe 이미지 랜드마크 z 는 손목 기준 상대 깊이이고 크기는 x 와 비슷한 배율이라고 문서에 적혀 있음 [열람, https://mediapipe.readthedocs.io/en/latest/solutions/hands.html]. span 을 sqrt(dx² + dy² + (W·dz)²) 로 계산해 단축을 되돌림. Iqbal 식의 근사 | 몇 줄 | 이미지 z 의 정확도는 공식 수치가 없음. z 잡음이 span 잡음으로 들어감 |
| S3 world 모양으로 단축 계수만 보정 | world 랜드마크의 손바닥 선분을 PnP 회전 R 로 카메라 좌표에 놓고, 시선에 수직인 성분 비율 c(단축 계수)를 구함. 거리 비 = (s₀/s)·(c/c₀). world 의 절대 크기는 비율에서 지워짐 | 반나절 | `m6` 의 PnP 회전 불안정(손바닥 뒤집힘)이 그대로 들어옴 [저장소] |
| S4 손가락 영향 제거 | span 점에서 1번(엄지 CMC)을 빼고 0, 5, 9, 13, 17 만 씀. 엄지 CMC 는 엄지를 맞댈 때 움직임 | 한 줄 | 점이 줄어 잡음이 조금 늘 수 있음 |

**검증할 수 있는 예측.** 회전이 원인이라면, S1~S3 로 보정한 span 에서는 가장 좋은 gain 이 0.5 보다 1 에 가까워져야 합니다. 같은 영상에서 gain 을 다시 훑으면 확인됩니다. gain 이 그대로 0.5 근처라면 원인은 회전이 아닌 다른 것(랜드마크 잡음, 손목과 손바닥 중심의 거리 차)입니다 [의견].

### 2.2 MediaPipe world landmark 의 크기와 깊이 정확도

- **world 좌표의 정의.** 단위는 미터이고 원점은 손의 대략적인 기하 중심입니다 [열람, MediaPipe Hands 문서]. Google 논문은 world 좌표를 중지 MCP 원점으로 정규화하고 크기는 유지하며, 화면 안 회전(roll)도 정규화한다고 씁니다 [열람, Sung 외 2021 2.1절].
- **크기는 측정이 아니라 사전 가정에 가깝습니다.** 정답을 만들 때 2D 주석에 GHUM 손 모델을 맞추면서 "사람 손 크기의 차이는 작고 그 차이는 모양에 반영된다"고 가정했다고 씁니다. 같은 모양의 큰 손과 작은 손은 원근 투영에서 구별되지 않는다는 이유입니다 [열람, 같은 절]. 보고된 평균 예측 오차는 1.5 cm 입니다(무엇의 평균인지 세부 정의는 논문에 짧게만 있음) [열람].
- **이 저장소의 결과와 맞습니다.** `m5_hand_scale.md` 는 world 손바닥 크기가 자세에 따라 흔들리고(IQR 왼 25.2 mm, 오른 12.6 mm), 그 크기와 PnP/센서 비의 상관이 약 −0.57 이라고 했습니다 [저장소]. world 크기가 학습된 사전 가정이라면, 그 출렁임은 모델이 자세를 크기로 잘못 해석한 결과로 볼 수 있습니다 [의견]. 그래서 **world 의 절대 크기는 거리 단서로 쓰지 말고, 쓴다면 모양과 방향(S3 의 단축 계수)만 쓰는 것이 안전합니다** [의견].
- 외부 검증: GMH-D(Amprimo 외 2023)는 RGB-D 깊이로 보강한 MediaPipe 가 기본 MediaPipe 보다 공간 측정이 정확했다고 보고합니다 [열람, 초록, https://arxiv.org/abs/2308.01088 (2023)]. world 크기의 mm 단위 오차를 거리별로 잰 독립 연구는 이번에 찾지 못했습니다.

### 2.3 RGB-D 깊이 공백의 원인과 RealSense D4xx 설정

**원인(공식 문서) [열람, RealSense "Tuning depth cameras for best performance", https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/ (페이지 2026 갱신, 원문은 D415/D435 기준)].**

| 원인 | 문서 내용 | 이 저장소에 대한 뜻 |
|---|---|---|
| 최소 거리 MinZ | MinZ 는 깊이 영상의 가로 해상도에 비례. 해상도를 낮추면 MinZ 가 줄어듦 | 424x240 에서 0.163 m, 640x480 에서 0.296 m [저장소: `m5` 3장]. 지금 녹화 영상의 손 거리 p5 는 0.29 m 근처라 424x240 에서는 MinZ 공백이 거의 없음 [저장소: `m7` 5장 2번] |
| 초점 | D4xx 는 50 cm~무한대에 초점이 맞춰져 있어 20 cm 보다 많이 가까우면 흐려짐 | 손이 0.2 m 보다 가까우면 MinZ 와 상관없이 품질이 떨어짐 |
| 가까운 입체물의 시점 차 | 아주 가까운 평평하지 않은 물체는 왼쪽과 오른쪽 카메라가 서로 다른 면을 봐서 정합이 안 됨(눈앞 손가락 예) | 0.2 m 근처 손가락과 손 가장자리 |
| 노출 | 나쁜 노출이 성능 저하의 첫째 원인. 게인은 낮게(16) | 실내 조명 확인 필요 |
| 레이저 포화 | 레이저 점이 국소 포화되면 레이저 출력을 낮추라고 권함 | 가까운 손바닥(반사)에서 생길 수 있음 [의견] |
| 반복 무늬 | 두 번째 정합 점수 문턱(DSSecondPeakThreshold) | 손에는 해당 적음 [의견] |

**공백 자체를 줄이는 설정 [열람, 같은 문서와 "D400 Series Visual Presets", https://dev.realsenseai.com/docs/d400-series-visual-presets/].**

- **Visual preset.** High Density 는 채움 비율이 높고 홀이 적습니다. High Accuracy 는 신뢰도 문턱이 높아 홀이 많습니다. Hand preset 은 손 추적과 제스처용이고 가장자리가 좋다고 적혀 있습니다. 이 저장소는 preset 을 기록만 하고 설정하지 않습니다(`tools/record_ego_depth.py` 113행) [저장소]. 어느 preset 으로 녹화됐는지는 `meta.json` 에 있습니다.
- **Disparity shift.** 값을 올리면 MinZ 가 줄고 MaxZ 는 더 빨리 줄어듭니다. 128 이면 새 MaxZ 가 원래 MinZ 와 비슷하고 새 MinZ 가 원래의 절반쯤이라고 씁니다. 이 저장소 기선 95 mm, 탐색 폭 126 px 를 쓰면(저장소의 식 `fx·0.095/126` 을 확장한 가정) 다음과 같습니다 [계산].

  | 깊이 해상도 | DS 0 | DS 20 | DS 30 | DS 50 |
  |---|---|---|---|---|
  | 424x240 (fx 216.5) MinZ / MaxZ | 0.163 m / 무한 | 0.141 / 1.03 m | 0.132 / 0.69 m | 0.117 / 0.41 m |
  | 640x480 (fx 392) MinZ / MaxZ | 0.296 m / 무한 | 0.255 / 1.86 m | 0.239 / 1.24 m | 0.212 / 0.74 m |

  손이 0.2~0.6 m 라면 424x240 에서 DS 20 이 MinZ 를 0.14 m 로 내리면서 1 m 안쪽을 지킵니다. 하지만 0.2 m 아래는 초점과 시점 차 문제가 남으므로 이득은 작을 것으로 봅니다 [의견].
- **SDK 후처리.** 0 을 빼고 시간 평균하는 IIR 시간 필터와, 마지막 유효값을 일정 시간 유지하는 "persistence"를 권합니다 [열람, 같은 문서]. persistence 는 화소 단위의 v3 hold 와 같은 생각입니다.
- 설정 변경의 효과는 실제 카메라로 손바닥 유효 화소 비율과 자연 공백 수를 재야 압니다. 이 문서는 재지 않았습니다.

### 2.4 공백 복귀 때 계단 줄이기와 지연 손해

**선행.**
- Ferstl 외 2021 의 fade(복귀 후 관절각을 시간에 걸쳐 보간)는 사람이 느끼는 오류를 줄였습니다 [열람, 초록 수준. fade 의 세부와 보간 시간은 04 문서 [저장소]].
- 칼만 측정 게이팅: 정규화 혁신 제곱(NIS)이 문턱보다 작은 측정만 받는 방식은 혁신 통계를 줄어든 쪽으로 치우치게 만든다는 분석이 있습니다 [열람, Or, https://arxiv.org/abs/2512.18508 (2025-12)]. 복귀 직후 큰 혁신을 게이트로 버리면 필터가 센서로 돌아오지 못하는 경우를 따로 막아야 합니다 [의견].
- 신호 상태에 따라 측정 잡음 R 을 키워 필터가 예측으로 버티게 하고, 상황이 바뀌면 한 주기 안에 다시 연다는 GNSS 쪽 예가 있습니다 [검색, https://arxiv.org/html/2610.41498 (2026)].
- 손 궤적 예측(Gamage 외 2021)은 추적 실패 때 궤적을 이어 주지만, 움직임 방향이 갑자기 바뀌면 오차가 크다고 보고합니다 [검색, https://3dvar.com/Gamage2021So.pdf (2021). PDF 라 열지 못함].

**세 방식의 지연 손해 [계산, 1차 모델 가정].** 복귀 첫 프레임의 계단을 Δ, 손의 시선 방향 속도를 v, 시간 상수를 τ 라 둡니다.

| 방식 | 계단 | 복귀 후 지연 손해 |
|---|---|---|
| (i) 출력 지수 블렌딩: 출력이 센서값을 τ 로 따라감 | e^(−t/τ) 로 줄어듦 | 손이 움직이면 정상 상태 뒤처짐 ≈ v·τ. v = 0.2 m/s, τ = 0.15 s 이면 30 mm. v4 의 `depth 단계 2s` 끝 계단 p50 22.5~23.3 mm [저장소]보다 큼 |
| (ii) 칼만 R 을 복귀 직후 키우고 τ 로 되돌림 | 위와 비슷 | 위와 같은 종류. R 비율에 따라 τ 가 정해짐 |
| (iii) 오프셋 감쇠: 출력 = 센서값 + (복귀 첫 프레임의 hold 추정 − 센서값)·e^(−t/τ) | 계단이 τ 에 걸쳐 퍼짐 | 센서 속도는 바로 따라가므로 v·τ 뒤처짐이 없음. 남는 오차는 Δ·e^(−t/τ) 뿐. Δ = 23 mm, τ = 0.15 s 이면 추가 최대 속도 ≈ Δ/τ = 0.15 m/s(한 프레임에 23 mm 튀면 약 0.7 m/s) |

(iii) 은 Ferstl 의 fade 와 같은 생각이고, 팔 쪽 클러치 재고정(04 문서 Isaac Teleop [저장소])과도 같습니다. 계단의 크기 Δ 는 그대로 정답에서 벗어나 있으므로, 공백 중 추정이 틀렸던 만큼 τ 동안 틀린 위치에 머뭅니다 [의견].

---

## 3. USB 2.1 문제

### 3.1 USB 3 포트인데 USB 2 로 붙는 알려진 원인

| 원인 | 근거 | 확인 수준 |
|---|---|---|
| 케이블(길이, 품질). 짧은 순정 케이블을 쓰는지 먼저 물어봄. Viewer 의 카메라 이름 옆 "2.1" 은 USB 3 포트에서 USB 2 로 잘못 붙었다는 표시 | librealsense #6992(D455, 2020) | [열람] |
| Type-C 쪽을 천천히 꽂으면 USB 2 로 열거. 다시 꽂으면 3.2. C-to-C 케이블에서는 재현 안 됨 | librealsense #2311(D400, 2018) | [열람] |
| 같은 포트에서 뽑았다 다시 꽂으면 3.2 로 고쳐짐 | librealsense #2052(D435, 2018) | [열람] |
| 다른 5 m 케이블에서 "USB 2.1 포트로 인식" | librealsense #9337(D455, 2021) | [열람] |
| 카메라 쪽 Type-C 커넥터를 뒤집어 꽂으면 해결 | librealsense #9381(2021). 단 이 이슈의 카메라는 L515 | [열람] |
| D457 은 USB 를 디버그용으로만 지원, USB 전용 카메라가 아님 | RealSense 포럼(2022) | [열람] |
| USB 2.1 에서 Motion Module 을 껐다 켜야 RGB/깊이가 시작되는 사례 | librealsense #12645(D455, 2024) | [열람] |

**이 저장소에서의 해석 [의견].** 같은 카메라가 2026-10-20 재설치 뒤 USB 3.2 로 붙어 컬러 + 깊이 640x480@30 이 30.0 fps 로 나왔습니다 [저장소: `r5` 머리말]. 그러므로 장치 고장보다 연결 상태(케이블, 꽂는 방식, 허브, 커넥터 방향)가 원인일 가능성이 큽니다. D457 이 USB 를 디버그용으로만 지원한다는 점은 연결이 다른 D400 보다 덜 안정적일 수 있다는 정황입니다.

**확인 방법(총괄 에이전트가 사용자 승인 뒤 실행, 이 문서는 실행하지 않음).**
- `lsusb -t` 에서 카메라 줄의 속도가 `5000M` 이상이면 USB 3, `480M` 이면 USB 2 [의견: 일반적인 Linux 동작].
- `dmesg` 의 연결 메시지가 "SuperSpeed" 인지 "high-speed" 인지 [의견].
- `rs-enumerate-devices` 의 USB 형식 줄, 또는 이 저장소 코드가 이미 쓰는 `camera_info.usb_type_descriptor`(`vision/rgbd_source.py` 94행, `tools/record_ego_depth.py` 62행) [저장소].
- 바꿔 볼 것: 짧은 케이블, 카메라 쪽을 빠르고 끝까지 꽂기, 커넥터 뒤집기, 허브 없이 노트북 포트에 바로, C-to-C 케이블.

### 3.2 USB 2 에서 쓸 수 있는 해상도와 fps

**Intel(현 RealSense) 공식 목록 [열람, "USB2 Support for RealSense Technology", https://www.realsenseai.com/news-insights/usb2-support-for-intel-realsense-technology/ (페이지 2026 갱신, 원래 발표는 FW 5.08 무렵)].**

| 해상도 | fps |
|---|---|
| 1280x720 | 6 |
| 640x480 | 6, 15, 30 |
| 480x270 | 6, 15, 30, 60 |

- 이 목록은 스트림 하나 기준인지 동시 사용 기준인지 밝히지 않습니다. 컬러와 깊이를 함께 쓸 수 있는 조합의 공식 표는 이번에 찾지 못했습니다.
- 같은 글은 USB 2 에서 센서 MIPI 읽기 시간이 USB 3.1 대비 약 4배 늘고, 글로벌 셔터(D435 등)는 버스트 읽기 때문에 문제가 될 수 있다고 씁니다. USB 2 에서는 펌웨어 갱신도 안 됩니다 [열람].
- 포럼 답변 요약: USB 2 에서 깊이와 RGB 를 640x480 에서 30 fps 까지 동시에 쓸 수 있다고 함 [검색, https://support.realsenseai.com/hc/en-us/community/posts/360052019014 (2021). 답변 본문은 열리지 않음].

**대역폭 [계산, 가정: 컬러 YUYV 2바이트/화소, 깊이 Z16 2바이트, 압축 없음].** USB 2 의 이론 속도는 480 Mb/s(60 MB/s)이고 실제 UVC 처리량은 이보다 꽤 낮습니다 [의견].

| 조합 | MB/s |
|---|---|
| 컬러 640x480@30 | 18.4 |
| 깊이 848x480@30 | 24.4 |
| 깊이 640x480@30 | 18.4 |
| 깊이 480x270@30 | 7.8 |
| 깊이 424x240@30 | 6.1 |
| 컬러 640x480@30 + 깊이 848x480@30 | 42.8 |
| 컬러 640x480@30 + 깊이 480x270@30 | 26.2 |
| 컬러 640x480@15 + 깊이 480x270@15 | 13.1 |

### 3.3 05 문서 4장 측정과 비교

| 05 문서의 USB 2 측정 [저장소] | 공식 목록과의 관계 |
|---|---|
| 컬러 640x480@30 단독 26 fps | 공식 목록에 있는 모드. 30 이 안 나온 것은 읽기 시간 4배와 노출 영향일 수 있음 [의견] |
| 깊이 480x270@30 단독 27.7 fps | 공식 목록에 있는 모드 |
| 컬러 + 깊이 15 fps → 11.2 fps | 해상도가 05 문서에 적혀 있지 않아 비교 불가. 15 fps 모드는 목록에 있음 |
| 컬러 640x480@30 + 깊이 848x480@30 시작 불가 | **848x480 은 USB 2 공식 목록에 없음.** 대역폭(42.8 MB/s)을 따지기 전에 지원 모드가 아님. 측정과 맞음 |

**결론.** USB 2 에서 컬러 + 깊이 30 fps 가 시작되지 않은 것은 공식 목록과 맞습니다(848x480 은 USB 2 모드가 아님). 공식 목록 안에서 동시 사용을 시도할 후보는 컬러 640x480 + 깊이 480x270 입니다(@15 는 13 MB/s, @30 은 26 MB/s). 05 문서의 11.2 fps 측정으로 보아 @30 동시는 어려울 가능성이 큽니다 [의견]. 또 저장소 코드는 USB 3 가 아니면 바로 종료하므로(`vision/rgbd_source.py` 94~96행) USB 2 에서는 시도조차 하지 않습니다 [저장소]. 가장 싼 해결은 3.1절의 연결 점검으로 USB 3.2 를 다시 얻는 것입니다. USB 2 를 꼭 써야 하면 06 문서의 "왼쪽 IR(Y8) + 깊이" 안(480x270@30 에서 Y8 3.9 + Z16 7.8 MB/s [계산])이 대역폭상 가장 가볍습니다.

---

## 4. 이 저장소에 바로 적용할 수 있는 것과 비용

비용은 `vision/depth_wrist_v4.py` 와 `tools/eval_depth_wrist_v4.py` 기준 추정입니다 [의견]. "오프라인"은 기존 npz 로 로봇과 카메라 없이 시험할 수 있다는 뜻입니다.

| 항목 | 근거 | 비용 | 오프라인 | 기대 효과 / 위험 |
|---|---|---|---|---|
| A 복귀 오프셋 감쇠 (2.4 iii) | Ferstl fade [열람, 초록], 2.4절 계산 | 몇 줄, 반나절 | 예 | 끝 계단의 프레임당 크기 감소, 움직임 뒤처짐 없음 / 틀린 추정에 τ 동안 머묾 |
| B 회전에 강한 span (S1, S2, S4) | Iqbal 2018, Sung 2021 [열람] | 몇 줄씩, 하루 | 예(효과 판정에는 회전이 많은 새 영상이 필요) | gain 을 1 쪽으로 돌릴 수 있음 / 이미지 z 의 정확도 미지 |
| C world 모양 단축 보정 (S3) | Iqbal 2018, MediaPipe world 정의 [열람] | 반나절 | 예 | 회전 보정 / PnP 회전 불안정 [저장소: m6] |
| D USB 3 연결 점검 | librealsense 이슈들 [열람] | 수 분, 사람 손 필요 | 아니오 | 컬러 + 깊이 30 fps 회복 / 없음 |
| E Visual preset(High Density, Hand), 레이저 출력, DS 20 | RealSense 문서 [열람] | 설정 몇 줄 + 녹화 하나 | 아니오(카메라 필요) | 공백 자체 감소 / 정확도 저하, 잘못된 깊이 증가 가능 |
| F EgoHand 비교 | 1장 a1~a7 [열람] | 설치와 가중치, 사용자 승인 필요 | 예 | 센서 없는 손목 깊이 기준선 / 이 노트북 GPU 지연 미확인 |

---

## 5. 다음에 시험할 순위 3개

### 1순위: 복귀 오프셋 감쇠(A)를 v4 위에

- **근거.** v4 의 남은 끝 계단 p50 은 `depth 단계 2s` 에서 22.5~23.3 mm 이고 p95 는 39.5~48.9 mm 입니다 [저장소: `m8` 4.1]. A 는 공백 중 추정의 정확도와 상관없이 그 계단이 팔로 가는 속도를 줄입니다. 출력 블렌딩(i)과 달리 움직임을 뒤처지지 않으므로 지연 비용이 작습니다(2.4절 계산). 기존 평가 틀에 그대로 넣을 수 있고 오프라인입니다.
- **지표.** 기존 "끝 계단"과 "끝 뒤 0.5 s 최대 계단"에 더해, 복귀 후 0.5 s 동안의 기준 대비 오차 적분(지연 손해)을 같은 표에 넣습니다. τ = 0.1 / 0.15 / 0.3 s 를 훑되, gain 처럼 이 영상에 맞춘 값이라고 적습니다.

### 2순위: 회전에 강한 span(B: S1, S2, S4)과 gain 다시 훑기

- **근거.** v4 의 알려진 한계 두 개(gain 을 영상 하나로 고름, 회전과 손가락 굽힘)를 직접 겨냥합니다. Iqbal 2018 은 상대 깊이와 뼈 길이 하나로 단축을 정확히 보정하는 식을 주고, Sung 2021 은 한 방향 2D 길이가 정면 손에서 불안정하다고 보고합니다 [열람]. 2.1절의 예측(보정하면 가장 좋은 gain 이 1 쪽으로 이동)을 지금 npz 로 바로 시험할 수 있습니다.
- **한계.** 지금 영상은 회전이 적을 수 있습니다 [저장소: `m8` 5장 3번]. 효과 판정에는 손바닥을 돌리는 구간이 있는 새 녹화가 필요하며, 녹화는 총괄이 사용자 승인을 받아 합니다.

### 3순위: 연결 점검(D) 후 깊이 설정(E)으로 공백 자체 줄이기

- **근거.** 실제 최소 거리 공백이 담긴 영상은 아직 없습니다 [저장소: `m7` 5장 2번, 9장]. 손이 0.2 m 근처까지 오는 영상을 preset 두세 개(지금 값, High Density, Hand)로 녹화하면, 손바닥 유효 화소 비율과 자연 공백 수를 비교할 수 있습니다. 이 녹화는 1, 2순위 변형의 실제 공백 평가에도 쓰입니다. 컬러 + 깊이 30 fps 가 필요하므로 USB 3 회복이 먼저입니다.
- **비용.** 카메라와 사람 손이 필요합니다. 설정은 `rs.option.visual_preset` 한 줄과 advanced mode 의 disparity shift 입니다.

### v1 순위와 다른 이유

| v1 | v2 | 이유 |
|---|---|---|
| 1순위 span 비율 | 구현됨(v4) | `m8_gap_v4.md` |
| 2순위 복귀 완화(변화율 제한 + σ_ray) | 1순위, 방식을 오프셋 감쇠로 구체화 | 2.4절 계산에서 출력 블렌딩과 σ 키우기는 움직이는 손에서 v·τ 만큼 뒤처지고, 그 크기가 v4 의 남은 계단과 비슷함 |
| 3순위 감쇠 속도 외삽 | 뺌 | 근거였던 Ferstl 의 "운동량 예측 불리" 문장을 확인하지 못함. v4 가 이미 앞뒤 움직임을 span 으로 따라가므로 겹침. Gamage 2021 은 방향이 바뀔 때 예측 오차가 크다고 보고 [검색] |
| 없음 | 2순위 회전 강건 span | v4 의 새 한계(gain, 회전)가 생겨서 |
| 없음 | 3순위 연결 점검과 깊이 설정 | USB 2.1 문제가 다시 생겼고, 공백 원인을 센서 쪽에서 줄이는 길은 v1 에 없었음 |

---

## 6. 확인하지 못한 것

- Ferstl 외 2021 본문의 운동량 예측 문장(PDF 를 열지 못함).
- EgoHand 표 1 의 WiLoR, HaWoR 행(잘림).
- Sensors 26(12) 3730 본문(403).
- USB 2 에서 컬러와 깊이를 동시에 쓸 수 있는 조합의 Intel 공식 표. 공식 글은 모드 목록만 줍니다.
- D455/D457 전용 MinZ 공식 수치와 disparity search 폭. 2.3절 표는 저장소의 `fx·0.095/126` 식을 확장한 계산입니다. 포럼 답변에는 D455 최소 거리가 약 0.5 m(최대 해상도 기준으로 보임)라는 요약이 있습니다 [검색, https://support.realsenseai.com/hc/en-us/community/posts/34029661511571 (2024)].
- Hand preset 의 D455 용 설정 내용과 효과 수치.
- MediaPipe 이미지 z 와 world 크기의 거리별 mm 오차를 잰 독립 연구.
- Gamage 외 2021 본문(PDF).
- 이 노트북 GPU 에서 EgoHand 의 지연.

---

## 7. 출처

**원문을 연 것 [열람].**

| 자료 | URL | 연도 |
|---|---|---|
| EgoHand 초록, HTML 표 1, 2, 5, 부록 A | https://arxiv.org/abs/2610.30293, https://arxiv.org/html/2610.30293 | 2026 |
| WiLoR README | https://github.com/rolpotamias/WiLoR | 2024 논문, README 2026 |
| HaWoR README | https://github.com/ThunderVVV/HaWoR | 2025 |
| HaMeR LICENSE.md | https://raw.githubusercontent.com/geopavlakos/hamer/main/LICENSE.md | 2023 |
| Ferstl 외 초록(Semantic Scholar API) | https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3474451.3476235 | 2021 |
| EgoVLA 초록, HTML 2~3장 | https://arxiv.org/abs/2507.12440 | 2025 |
| EgoScale 초록, 페이지 | https://arxiv.org/abs/2602.16710, https://research.nvidia.com/labs/gear/egoscale/ | 2026 |
| Dexterous Point Policy 초록 | https://arxiv.org/abs/2606.10614 | 2026 |
| SpatialDex 초록 | https://arxiv.org/abs/2610.29716 | 2026 |
| Motion Tracks(MT-π) 초록 | https://arxiv.org/abs/2501.06994 | 2025 |
| InfiniHand 초록 | https://arxiv.org/abs/2609.35743 | 2026 |
| FounRef 초록 | https://arxiv.org/abs/2609.29224 | 2026 |
| MOMA 초록 | https://arxiv.org/abs/2506.17110 | 2025 |
| IMM-MOT 초록 | https://arxiv.org/abs/2502.09672 | 2025 |
| Motion Dynamics KF 초록 | https://arxiv.org/abs/2505.07254 | 2025 |
| Mueller 외 초록 | https://arxiv.org/abs/1704.02201 | 2017 |
| Iqbal 외, 2.5D 히트맵, 3장 | https://ar5iv.labs.arxiv.org/html/1804.09534 | 2018 |
| Sung 외, On-device Real-time Hand Gesture Recognition, 2.1절 | https://arxiv.org/html/2111.00038 | 2021 |
| MediaPipe Hands 문서(world landmark 정의, 이미지 z) | https://mediapipe.readthedocs.io/en/latest/solutions/hands.html | 문서 |
| MediaPipe Hand Landmarker 문서 | https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker | 문서 |
| GMH-D 초록 | https://arxiv.org/abs/2308.01088 | 2023 |
| Or, 게이팅된 칼만 필터의 혁신 통계 | https://arxiv.org/abs/2512.18508 | 2025 |
| RealSense Tuning depth cameras for best performance | https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/ | 문서(2026 갱신) |
| RealSense D400 Series Visual Presets | https://dev.realsenseai.com/docs/d400-series-visual-presets/ | 문서(2026 갱신) |
| RealSense USB2 Support | https://www.realsenseai.com/news-insights/usb2-support-for-intel-realsense-technology/ | 문서(2026 갱신) |
| librealsense #6992, #2311, #2052, #9337, #9381, #12645 | https://github.com/realsenseai/librealsense/issues/6992, https://github.com/IntelRealSense/librealsense/issues/2311, https://github.com/IntelRealSense/librealsense/issues/2052, https://github.com/IntelRealSense/librealsense/issues/9337, https://github.com/IntelRealSense/librealsense/issues/9381, https://github.com/IntelRealSense/librealsense/issues/12645 | 2018~2024 |
| RealSense 포럼, D457 USB compatibility in production | https://support.realsenseai.com/hc/en-us/community/posts/10293079498387 | 2022 |

**검색 요약만 본 것 [검색].** Sensors 26(12) 3730 https://www.mdpi.com/1424-8220/26/12/3730 (2026), Gamage 외 So Predictable! https://3dvar.com/Gamage2021So.pdf (2021), GNSS R 팽창 게이트 https://arxiv.org/html/2610.41498 (2026), RealSense 포럼 USB 2 동시 스트림 https://support.realsenseai.com/hc/en-us/community/posts/360052019014 (2021), RealSense 포럼 D455 최소 거리 https://support.realsenseai.com/hc/en-us/community/posts/34029661511571 (2024).

**열기를 시도했으나 실패한 것.** Ferstl 저자 PDF(PDF 형식 미지원), ACM DL 페이지(403), MDPI/PMC 본문(403), RealSense 포럼 USB 2 글(403), Semantic Scholar 검색 API(429).

**사용한 검색어.**
1. EgoHand NVIDIA egocentric 3D hand pose camera space OpenMDW arXiv
2. Ferstl McDonnell Neff 2021 hand tracking loss fade rest hybrid momentum prediction SAP
3. "Mitigating the Impact of Hand Tracking Loss" momentum predict trajectory not effective
4. Ferstl 2021 hand tracking loss avatar "momentum" extrapolation strategy perceived worse
5. Sensors 2026 26(12) 3730 MediaPipe ZED stereo depth hand landmarks median EMA bone length
6. monocular hand distance estimation from 2D keypoints palm size foreshortening rotation compensation MediaPipe
7. MediaPipe hand world landmarks accuracy metric scale evaluation depth error mm study
8. Intel RealSense D400 tuning best performance white paper disparity shift minimum distance MinZ High Density preset
9. librealsense D455 connected as USB 2.1 on USB 3 port issue github
10. RealSense D400 USB 2.0 supported stream profiles depth color simultaneous table 15 fps 424x240 640x480
11. librealsense USB2 mode D455 depth and color simultaneously fps limit github issue
12. D457 USB type-C mode GMSL USB 3 cable recognized USB 2 librealsense
13. dev.realsenseai.com USB 2 support D400 depth camera modes color 1280x720 6fps depth 640x480 15fps
14. dev.realsenseai.com tuning depth cameras best performance MinZ disparity shift lower resolution 848x480 decreases minimum range
15. Kalman filter measurement noise inflation after track reacquisition dropout re-entry smooth transition innovation gating
16. estimate hand distance from camera using palm width and palm length orientation invariant monocular webcam MediaPipe z depth

이 문서에는 IP 주소, 시리얼 번호, 회사 로고, 내부 문서 내용을 넣지 않았습니다.
