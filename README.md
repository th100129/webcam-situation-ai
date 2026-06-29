# Focus Trigger — 전체 모듈화 버전

웹캠에서 **얼굴·손·자세·객체**를 동시에 분석하고, 집중 흐림이 일정 시간 유지되면 **경고음 + 랜덤 미션**을 실행하는 프로젝트입니다.

## 포함된 기능

- Face Landmarker 기반 얼굴 메쉬 표시
- 눈 감김(EAR) 판정: **5초 이상** 유지 시 집중 흐림 신호
- 얼굴 방향 이탈, 고개 숙임, 얼굴 미감지, 손-얼굴 접촉 판정
- Pose Landmarker 기반 어깨 기울기·움직임·양손 들기 판정
- Hand Landmarker 기반 손바닥 / 주먹 / 엄지척 / 브이 / 검지 가리키기 인식
- YOLO 객체 인식 및 안정 객체 추적
- 집중 흐림 점수와 시간 기반 안정화 로직
- 집중 흐림 시 경고음과 랜덤 미션 실행
- **미션 성공 전까지 경고음 유지**
- 제한 시간 초과는 기록만 남기고 같은 미션 재도전
- 성공 시 파티클 효과, 성공 피드백, 쿨다운
- JSONL 이벤트 로그 + 세션 종료 리포트(JSON)
- OpenCV 한글 깨짐을 피하기 위한 Pillow + 맑은 고딕 렌더링

## 폴더 구조

```text
focus-trigger-full-modular/
├─ main.py                         # 전체 실행 흐름
├─ config.py                       # 임계값 / 카메라 / 모델 경로
├─ download_models.py              # Face·Hand·Pose task 모델 자동 다운로드
├─ setup.bat / run.bat
│
├─ detectors/
│  ├─ face_mesh_detector.py        # 얼굴 메쉬, 눈 감김, 얼굴 방향, 고개 숙임
│  ├─ hand_detector.py             # 손 랜드마크, 손가락 수, 제스처
│  ├─ pose_detector.py             # 자세 흔들림, 양손 들기
│  └─ object_detector.py           # YOLO 객체 감지
├─ focus/
│  ├─ focus_analyzer.py            # 집중 흐림 점수와 유지 시간 판정
│  └─ focus_action_manager.py      # 트리거·경고음·쿨다운 상태 관리
├─ mission/
│  └─ mission_manager.py           # 랜덤 미션 선택 / 성공 판정 / 재도전
├─ feedback/
│  ├─ llm_manager.py               # 기본 문구 또는 선택적 LLM API 문구
│  └─ prompt_templates.py
├─ tracking/
│  └─ object_state_tracker.py
├─ ui/
│  ├─ renderer.py                  # 한글 UI, 미션 오버레이
│  └─ particle_system.py           # 성공 파티클
└─ utils/
   ├─ camera.py
   ├─ screenshot.py
   ├─ session_logger.py
   ├─ metrics_tracker.py
   └─ report_generator.py
```

## 가장 쉬운 실행 방법

압축을 풀고, **main.py가 보이는 폴더를 VS Code로 열어야 합니다.**

### 방법 1 — 배치 파일

1. `setup.bat` 더블 클릭: 라이브러리 설치와 MediaPipe 모델 다운로드를 한 번에 실행
2. 끝나면 `run.bat` 더블 클릭

### 방법 2 — VS Code PowerShell

```powershell
python -m pip install -r requirements.txt
python download_models.py
python main.py
```

## 조작법

| 키 | 기능 |
|---|---|
| `Q` / `ESC` | 종료 |
| `S` | 스크린샷 저장 (`captures/`) |
| `D` | 얼굴·손·자세 랜드마크 선/점 표시 켜기·끄기 |

## 미션 목록

| 미션 ID | 성공 조건 |
|---|---|
| `open_palm` | 손바닥을 펼친 상태 유지 |
| `fist` | 주먹 상태 유지 |
| `thumbs_up` | 엄지척 상태 유지 |
| `v_sign` | 브이 상태 유지 |
| `pointing` | 검지만 편 상태 유지 |
| `look_forward` | 정면 + 눈 뜸 + 고개 들기 상태 유지 |
| `hold_still` | 자세 흔들림 없이 유지 |
| `hands_up` | 양 손목이 어깨보다 위에 있도록 유지 |

## 조절이 필요한 값

`config.py`에서 바꿉니다.

```python
EYE_CLOSED_EAR_THRESHOLD = 0.205  # 안경 때문에 눈 감김이 너무 잘못 잡히면 0.19~0.20으로 낮추기
EYES_CLOSED_SECONDS = 5.0         # 눈 감김 트리거 시간
MISSION_TIMEOUT_SECONDS = 8.0     # 재도전 안내가 표시되는 간격
OBJECT_DETECTION_INTERVAL = 3     # 높이면 YOLO 실행이 줄어 FPS가 오름
```

## LLM API는 기본적으로 꺼져 있음

기본 설정은 인터넷/API 키 없이도 작동하는 템플릿 문구입니다.

실제 OpenAI API 문구를 쓰고 싶을 때만:

1. 환경변수 `OPENAI_API_KEY` 설정
2. `config.py`에서 `USE_LLM_API = True`

## Git 적용 권장 순서

기존의 잘못된 `webcam-situation-ai_modularized` 폴더와 섞지 말고, 이 폴더에서 먼저 실행 테스트를 합니다.

정상 실행이 확인되면 기존 Git 저장소의 `dev` 브랜치에 이 폴더 내용을 복사한 뒤:

```powershell
git add .
git commit -m "refactor: modularize full focus mission system"
git push origin dev
```
