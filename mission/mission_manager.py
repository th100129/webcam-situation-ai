from __future__ import annotations

import random
import time
from typing import Dict, Iterable, List

from config import (
    MISSION_GESTURE_HOLD_SECONDS,
    MISSION_HANDS_UP_HOLD_SECONDS,
    MISSION_LOOK_FORWARD_HOLD_SECONDS,
    MISSION_STILL_HOLD_SECONDS,
    MISSION_TIMEOUT_SECONDS,
)
from core.types import HandInfo, PoseInfo


MISSION_LIBRARY: Dict[str, Dict] = {
    "open_palm": {
        "title": "손바닥 보여주기",
        "instruction": "카메라를 향해 손바닥을 펼쳐주세요.",
        "hold_seconds": MISSION_GESTURE_HOLD_SECONDS,
    },
    "fist": {
        "title": "주먹 쥐기",
        "instruction": "손을 가볍게 주먹으로 쥐어주세요.",
        "hold_seconds": MISSION_GESTURE_HOLD_SECONDS,
    },
    "thumbs_up": {
        "title": "엄지척 하기",
        "instruction": "엄지손가락을 위로 들어 올려주세요.",
        "hold_seconds": MISSION_GESTURE_HOLD_SECONDS,
    },
    "v_sign": {
        "title": "브이 하기",
        "instruction": "검지와 중지로 브이 동작을 해주세요.",
        "hold_seconds": MISSION_GESTURE_HOLD_SECONDS,
    },
    "pointing": {
        "title": "검지 가리키기",
        "instruction": "검지만 펴서 카메라 쪽으로 보여주세요.",
        "hold_seconds": MISSION_GESTURE_HOLD_SECONDS,
    },
    "look_forward": {
        "title": "정면 바라보기",
        "instruction": "화면을 보며 정면을 3초간 유지해주세요.",
        "hold_seconds": MISSION_LOOK_FORWARD_HOLD_SECONDS,
    },
    "hold_still": {
        "title": "가만히 있기",
        "instruction": "어깨를 편하게 두고 3초간 가만히 있어주세요.",
        "hold_seconds": MISSION_STILL_HOLD_SECONDS,
    },
    "hands_up": {
        "title": "양손 들기",
        "instruction": "양손을 어깨보다 위로 잠깐 들어주세요.",
        "hold_seconds": MISSION_HANDS_UP_HOLD_SECONDS,
    },
}

REASON_MISSIONS = {
    "eyes_closed": ["open_palm", "thumbs_up", "look_forward", "v_sign"],
    "face_missing": ["open_palm", "thumbs_up", "v_sign", "look_forward"],
    "head_down": ["look_forward", "thumbs_up", "open_palm", "hold_still"],
    "face_away": ["look_forward", "v_sign", "open_palm"],
    "hand_face_contact": ["fist", "v_sign", "thumbs_up", "hold_still"],
    "pose_unstable": ["hold_still", "hands_up", "open_palm"],
}


class MissionManager:
    """미션을 선택하고, 성공 전까지 같은 미션을 유지하며 성공 여부를 판정한다."""

    def __init__(self) -> None:
        self.active = False
        self.mission_id: str | None = None
        self.reason: str | None = None
        self.started_at = 0.0
        self.attempt_started_at = 0.0
        self.match_started_at: float | None = None
        self.timeout_count = 0
        self.last_mission_id: str | None = None

    def start(self, reason: str | None) -> Dict:
        candidates = list(REASON_MISSIONS.get(reason, MISSION_LIBRARY.keys()))
        if len(candidates) > 1 and self.last_mission_id in candidates:
            candidates.remove(self.last_mission_id)
        self.mission_id = random.choice(candidates)
        self.last_mission_id = self.mission_id
        self.reason = reason or "unknown"
        now = time.monotonic()
        self.started_at = now
        self.attempt_started_at = now
        self.match_started_at = None
        self.timeout_count = 0
        self.active = True
        return self.current_result()

    def update(self, focus_analysis: Dict, hands: Iterable[HandInfo], poses: Iterable[PoseInfo]) -> Dict:
        if not self.active or self.mission_id is None:
            return self.empty_result()

        now = time.monotonic()
        hands = list(hands)
        poses = list(poses)
        matched = self._is_matched(focus_analysis, hands, poses)
        hold_seconds = MISSION_LIBRARY[self.mission_id]["hold_seconds"]

        if matched:
            if self.match_started_at is None:
                self.match_started_at = now
        else:
            self.match_started_at = None

        held_for = 0.0 if self.match_started_at is None else now - self.match_started_at
        progress = min(1.0, held_for / hold_seconds)
        completed = matched and held_for >= hold_seconds
        timed_out = False

        # 시간 안에 못 해도 경고음은 끄지 않는다. 같은 미션의 재도전만 기록한다.
        if not completed and now - self.attempt_started_at >= MISSION_TIMEOUT_SECONDS:
            timed_out = True
            self.timeout_count += 1
            self.attempt_started_at = now
            self.match_started_at = None
            progress = 0.0

        result = self.current_result(
            matched=matched,
            held_for=held_for,
            progress=progress,
            completed=completed,
            timed_out=timed_out,
        )
        if completed:
            self.active = False
            result["active"] = False
            result["completed"] = True
            result["reaction_time"] = now - self.started_at
        return result

    def _is_matched(self, focus_analysis: Dict, hands: List[HandInfo], poses: List[PoseInfo]) -> bool:
        assert self.mission_id is not None
        if self.mission_id in {"open_palm", "fist", "thumbs_up", "v_sign", "pointing"}:
            return any(hand.gesture == self.mission_id for hand in hands)
        if self.mission_id == "look_forward":
            return (
                focus_analysis["face"]["detected"]
                and focus_analysis["face"]["direction"] == "forward"
                and focus_analysis["face"]["eye_state"] == "open"
                and not focus_analysis["face"]["head_down"]
            )
        if self.mission_id == "hold_still":
            return bool(poses) and not poses[0].unstable
        if self.mission_id == "hands_up":
            return bool(poses) and poses[0].hands_up
        return False

    def current_result(
        self,
        matched: bool = False,
        held_for: float = 0.0,
        progress: float = 0.0,
        completed: bool = False,
        timed_out: bool = False,
    ) -> Dict:
        if not self.active or self.mission_id is None:
            return self.empty_result()
        mission = MISSION_LIBRARY[self.mission_id]
        now = time.monotonic()
        return {
            "active": True,
            "completed": completed,
            "timed_out": timed_out,
            "mission_id": self.mission_id,
            "mission_title": mission["title"],
            "instruction": mission["instruction"],
            "reason": self.reason,
            "elapsed": now - self.started_at,
            "attempt_elapsed": now - self.attempt_started_at,
            "hold_required": mission["hold_seconds"],
            "held_for": held_for,
            "progress": progress,
            "matched": matched,
            "timeout_count": self.timeout_count,
            "reaction_time": None,
        }

    @staticmethod
    def empty_result() -> Dict:
        return {
            "active": False,
            "completed": False,
            "timed_out": False,
            "mission_id": None,
            "mission_title": "",
            "instruction": "",
            "reason": None,
            "elapsed": 0.0,
            "attempt_elapsed": 0.0,
            "hold_required": 0.0,
            "held_for": 0.0,
            "progress": 0.0,
            "matched": False,
            "timeout_count": 0,
            "reaction_time": None,
        }

    @staticmethod
    def signature(result: Dict) -> tuple:
        return (
            result["active"],
            result["mission_id"],
            result["completed"],
            result["timed_out"],
            int(result["progress"] * 4),
        )
