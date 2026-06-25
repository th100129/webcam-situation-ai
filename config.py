# config.py

from pathlib import Path


# =========================
# 기본 경로
# =========================

LOG_DIR = Path("logs")
REPORT_DIR = Path("logs")


# =========================
# LLM 설정
# =========================

# 처음에는 False로 두는 게 안전함.
# True로 바꾸면 실제 LLM API 호출을 시도함.
USE_LLM_API = False

# OpenAI API를 나중에 쓸 경우 모델명만 여기서 관리
LLM_MODEL_NAME = "gpt-4o-mini"


# =========================
# 미션 설정
# =========================

MISSION_TIMEOUT_SECONDS = 8.0

MISSION_TYPES = [
    "open_palm",
    "fist",
    "two_fingers",
    "pointing",
]

MISSION_LABELS = {
    "open_palm": "손바닥 보여주기",
    "fist": "주먹 쥐기",
    "two_fingers": "브이 하기",
    "pointing": "검지로 가리키기",
}

MISSION_DESCRIPTIONS = {
    "open_palm": "손바닥을 카메라에 보여주세요.",
    "fist": "주먹을 쥐어주세요.",
    "two_fingers": "브이 동작을 보여주세요.",
    "pointing": "검지만 펴서 가리키는 동작을 보여주세요.",
}


# =========================
# 집중 흐림 사유
# =========================

REASON_LABELS = {
    "eyes_closed": "눈 감김",
    "head_down": "고개 숙임",
    "face_missing": "얼굴 미감지",
    "hand_face_contact": "손-얼굴 접촉",
    "unstable_pose": "자세 흔들림",
    "object_or_hand_change": "객체/손 상태 변화",
    "unknown": "알 수 없음",
}


# =========================
# 화면 표시 설정
# =========================

MAX_PANEL_MESSAGE_LENGTH = 70