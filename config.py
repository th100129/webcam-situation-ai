from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
CAPTURE_DIR = PROJECT_ROOT / "captures"

# ---------------------------------------------------------------------------
# 카메라 / 화면
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
# 카메라에는 1280x720을 먼저 요청한다. 지원하지 않으면 실제 가능한 해상도로 자동 fallback된다.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
# 기존 코드 호환용 별칭
FRAME_WIDTH = CAMERA_WIDTH
FRAME_HEIGHT = CAMERA_HEIGHT

# 감지는 이 크기 이하에서만 수행해 FPS를 지킨다. 화면 출력과 분리되어 있다.
ANALYSIS_MAX_WIDTH = 960
ANALYSIS_MAX_HEIGHT = 540
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
MIRROR_VIEW = True
WINDOW_NAME = "Focus Trigger - Webcam Focus Mission"

# 기본 화면은 깔끔하게, D 키를 눌렀을 때만 랜드마크 점/선이 많이 보인다.
DEBUG_LANDMARKS_DEFAULT = False

# ---------------------------------------------------------------------------
# 모델 파일
# ---------------------------------------------------------------------------
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
YOLO_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

# ---------------------------------------------------------------------------
# YOLO 객체 인식
# ---------------------------------------------------------------------------
ENABLE_OBJECT_DETECTION = True
OBJECT_DETECTION_INTERVAL = 3
YOLO_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 640
TARGET_CLASSES = None
APPEAR_THRESHOLD_SECONDS = 0.8
DISAPPEAR_THRESHOLD_SECONDS = 2.0

# ---------------------------------------------------------------------------
# MediaPipe 감지 설정
# ---------------------------------------------------------------------------
MAX_NUM_FACES = 1
MAX_NUM_HANDS = 2
MAX_NUM_POSES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# 실제 랜드마크 렌더링은 ui/renderer.py에서 통합한다.
DRAW_FACE_MESH = False
DRAW_HAND_LANDMARKS = False
DRAW_POSE_LANDMARKS = False

# ---------------------------------------------------------------------------
# 얼굴 / 자세 판정값
# ---------------------------------------------------------------------------
EYE_CLOSED_EAR_THRESHOLD = 0.205
FACE_AWAY_YAW_RATIO_THRESHOLD = 0.34
HEAD_DOWN_RATIO_THRESHOLD = 0.36
HAND_FACE_PADDING_RATIO = 0.15
POSE_TILT_THRESHOLD = 0.18
POSE_MOTION_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# 집중 흐림: 아래 시간 이상 유지될 때만 실제 미션을 트리거한다.
# ---------------------------------------------------------------------------
FACE_MISSING_SECONDS = 2.0
FACE_AWAY_SECONDS = 3.0
HEAD_DOWN_SECONDS = 3.0
EYES_CLOSED_SECONDS = 5.0
HAND_FACE_CONTACT_SECONDS = 3.0
POSE_UNSTABLE_SECONDS = 3.0

# ---------------------------------------------------------------------------
# 집중 흐림 액션 / 경고음
# ---------------------------------------------------------------------------
FOCUS_TRIGGER_HOLD_SECONDS = 0.8
FOCUS_COOLDOWN_SECONDS = 5.0
WARNING_BEEP_INTERVAL_SECONDS = 0.7
WARNING_BEEP_FREQUENCY = 880
WARNING_BEEP_DURATION_MS = 110
SOUND_TEST_LOW_FREQUENCY = 660
SOUND_TEST_HIGH_FREQUENCY = 880
SOUND_TEST_DURATION_MS = 180

# ---------------------------------------------------------------------------
# 미션
# ---------------------------------------------------------------------------
MISSION_TIMEOUT_SECONDS = 8.0
MISSION_GESTURE_HOLD_SECONDS = 0.7
MISSION_LOOK_FORWARD_HOLD_SECONDS = 3.0
MISSION_STILL_HOLD_SECONDS = 3.0
MISSION_HANDS_UP_HOLD_SECONDS = 0.8

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
USE_LLM_API = False
LLM_MODEL_NAME = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# 폰트
# ---------------------------------------------------------------------------
KOREAN_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
]
