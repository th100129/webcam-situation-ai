from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

from config import (
    EYES_CLOSED_SECONDS,
    FACE_AWAY_SECONDS,
    FACE_MISSING_SECONDS,
    HAND_FACE_CONTACT_SECONDS,
    HAND_FACE_PADDING_RATIO,
    HEAD_DOWN_SECONDS,
    POSE_UNSTABLE_SECONDS,
)
from core.types import FaceInfo, HandInfo, PoseInfo
from utils.geometry import point_in_bbox


@dataclass
class SignalState:
    started_at: float | None = None


class DurationTracker:
    """프레임 단위 감지를 시간 단위의 안정 신호로 바꾼다."""

    def __init__(self) -> None:
        self.states: Dict[str, SignalState] = {}

    def update(self, name: str, active: bool, now: float) -> float:
        state = self.states.setdefault(name, SignalState())
        if active:
            if state.started_at is None:
                state.started_at = now
            return now - state.started_at
        state.started_at = None
        return 0.0


class FocusAnalyzer:
    """얼굴·손·자세 신호를 모아 집중 흐림 점수와 사유를 계산한다."""

    SIGNAL_SECONDS = {
        "face_missing": FACE_MISSING_SECONDS,
        "face_away": FACE_AWAY_SECONDS,
        "head_down": HEAD_DOWN_SECONDS,
        "eyes_closed": EYES_CLOSED_SECONDS,
        "hand_face_contact": HAND_FACE_CONTACT_SECONDS,
        "pose_unstable": POSE_UNSTABLE_SECONDS,
    }

    SIGNAL_SCORES = {
        "face_missing": 60,
        "face_away": 45,
        "head_down": 45,
        "eyes_closed": 75,
        "hand_face_contact": 25,
        "pose_unstable": 25,
    }

    REASON_LABELS = {
        "face_missing": "얼굴 미감지",
        "face_away": "정면 이탈",
        "head_down": "고개 숙임",
        "eyes_closed": "눈 감김",
        "hand_face_contact": "손-얼굴 접촉",
        "pose_unstable": "자세 흔들림",
    }

    def __init__(self) -> None:
        self.tracker = DurationTracker()
        self.has_seen_face = False

    def analyze(
        self,
        faces: Iterable[FaceInfo],
        hands: Iterable[HandInfo],
        poses: Iterable[PoseInfo],
    ) -> Dict:
        now = time.monotonic()
        faces = list(faces)
        hands = list(hands)
        poses = list(poses)
        primary_face = faces[0] if faces else None
        primary_pose = poses[0] if poses else None

        if primary_face is not None:
            self.has_seen_face = True

        raw = {
            "face_missing": self.has_seen_face and primary_face is None,
            "face_away": primary_face is not None and primary_face.direction != "forward",
            "head_down": primary_face is not None and primary_face.head_down,
            "eyes_closed": primary_face is not None and primary_face.eyes_closed,
            "hand_face_contact": self._has_hand_face_contact(primary_face, hands),
            "pose_unstable": primary_pose is not None and primary_pose.unstable,
        }

        durations = {name: self.tracker.update(name, active, now) for name, active in raw.items()}
        stable = {
            name: raw[name] and durations[name] >= self.SIGNAL_SECONDS[name]
            for name in raw
        }
        active_reasons = [name for name, is_stable in stable.items() if is_stable]
        score = min(100, sum(self.SIGNAL_SCORES[name] for name in active_reasons))
        level = self._level_from_score(score)

        face_summary = {
            "detected": primary_face is not None,
            "direction": primary_face.direction if primary_face else "missing",
            "eye_state": "closed" if primary_face and primary_face.eyes_closed else "open" if primary_face else "missing",
            "ear": primary_face.ear if primary_face else 0.0,
            "head_down": bool(primary_face and primary_face.head_down),
        }
        pose_summary = {
            "detected": primary_pose is not None,
            "unstable": bool(primary_pose and primary_pose.unstable),
            "motion": primary_pose.motion if primary_pose else 0.0,
            "hands_up": bool(primary_pose and primary_pose.hands_up),
        }
        return {
            "score": score,
            "level": level,
            "raw_indicators": raw,
            "stable_indicators": stable,
            "durations": durations,
            "active_reasons": active_reasons,
            "active_reason_labels": [self.REASON_LABELS[name] for name in active_reasons],
            "face": face_summary,
            "pose": pose_summary,
        }

    @staticmethod
    def _has_hand_face_contact(face: FaceInfo | None, hands: List[HandInfo]) -> bool:
        if face is None:
            return False
        for hand in hands:
            # 손목, 검지 끝, 중지 끝이 얼굴 bbox 안 또는 근처에 들어오면 접촉 가능성으로 본다.
            # Face bbox는 픽셀 좌표이므로, 같은 정규화 좌표계의 얼굴 랜드마크로 영역을 다시 만든다.
            fx = [point[0] for point in face.landmarks]
            fy = [point[1] for point in face.landmarks]
            normalized_bbox = (min(fx), min(fy), max(fx), max(fy))
            for landmark_index in (0, 8, 12):
                x, y, _ = hand.landmarks[landmark_index]
                if point_in_bbox((x, y), normalized_bbox, padding=HAND_FACE_PADDING_RATIO):
                    return True
        return False

    @staticmethod
    def _level_from_score(score: int) -> str:
        if score >= 75:
            return "high"
        if score >= 50:
            return "warning"
        if score > 0:
            return "caution"
        return "normal"

    @staticmethod
    def signature(analysis: Dict) -> tuple:
        return (
            analysis["level"],
            tuple(analysis["active_reasons"]),
            analysis["face"]["direction"],
            analysis["face"]["eye_state"],
        )
