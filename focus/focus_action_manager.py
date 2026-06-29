from __future__ import annotations

import platform
import threading
import time
from typing import Dict, Optional

from config import (
    FOCUS_COOLDOWN_SECONDS,
    FOCUS_TRIGGER_HOLD_SECONDS,
    SOUND_TEST_DURATION_MS,
    SOUND_TEST_HIGH_FREQUENCY,
    SOUND_TEST_LOW_FREQUENCY,
    WARNING_BEEP_DURATION_MS,
    WARNING_BEEP_FREQUENCY,
    WARNING_BEEP_INTERVAL_SECONDS,
)


class FocusActionManager:
    """집중 흐림 감지, 경고음, 미션 성공 뒤 쿨다운을 관리한다."""

    PRIORITY = ["eyes_closed", "face_missing", "head_down", "face_away", "hand_face_contact", "pose_unstable"]

    def __init__(self) -> None:
        self.focus_blur_since: float | None = None
        self.cooldown_until = 0.0
        self.episode_triggered = False
        self.last_beep_time = 0.0
        self.last_sound_error: str | None = None
        self.sound_test_requested_at = 0.0

    def update(self, focus_analysis: Dict, mission_active: bool) -> Dict:
        now = time.monotonic()
        reasons = list(focus_analysis["active_reasons"])
        detected = bool(reasons)
        primary_reason = self._pick_primary_reason(reasons)

        result = {
            "state": "monitoring",
            "focus_blur_detected": detected,
            "should_start_mission": False,
            "primary_reason": primary_reason,
            "duration": 0.0,
            "cooldown_left": max(0.0, self.cooldown_until - now),
        }

        if mission_active:
            result["state"] = "mission_active"
            return result

        if now < self.cooldown_until:
            result["state"] = "cooldown"
            self.focus_blur_since = None
            self.episode_triggered = False
            return result

        if not detected:
            self.focus_blur_since = None
            self.episode_triggered = False
            return result

        if self.focus_blur_since is None:
            self.focus_blur_since = now
        duration = now - self.focus_blur_since
        result["duration"] = duration
        if duration < FOCUS_TRIGGER_HOLD_SECONDS:
            result["state"] = "watching"
            return result

        result["state"] = "focus_blur"
        if not self.episode_triggered:
            result["should_start_mission"] = True
            self.episode_triggered = True
        return result

    def mission_succeeded(self) -> None:
        self.cooldown_until = time.monotonic() + FOCUS_COOLDOWN_SECONDS
        self.focus_blur_since = None
        self.episode_triggered = False

    def maybe_play_warning(self, mission_active: bool) -> None:
        """미션이 활성인 동안 지정 간격으로 Windows 경고음을 요청한다."""
        if not mission_active:
            return
        now = time.monotonic()
        if now - self.last_beep_time < WARNING_BEEP_INTERVAL_SECONDS:
            return
        self.last_beep_time = now
        self._run_async(self._warning_beep_windows)

    def play_sound_test(self) -> tuple[bool, str]:
        """B 키용 사운드 테스트. 청취 여부는 사용자의 출력 장치 상태에 달려 있다."""
        self.sound_test_requested_at = time.monotonic()
        if platform.system() != "Windows":
            message = "현재 운영체제에서는 Windows 경고음 테스트를 지원하지 않습니다."
            self.last_sound_error = message
            return False, message

        self.last_sound_error = None
        self._run_async(self._sound_test_windows)
        return True, "사운드 테스트 요청을 보냈습니다. 기본 스피커/이어폰에서 짧은 두 번의 알림음이 나야 합니다."

    def sound_status(self) -> str:
        if self.last_sound_error:
            return f"경고음 오류: {self.last_sound_error}"
        return "경고음 준비됨"

    def _run_async(self, target) -> None:
        threading.Thread(target=target, daemon=True).start()

    def _warning_beep_windows(self) -> None:
        if platform.system() != "Windows":
            return
        try:
            import winsound

            winsound.Beep(WARNING_BEEP_FREQUENCY, WARNING_BEEP_DURATION_MS)
        except Exception as error:
            self.last_sound_error = str(error)
            print(f"[경고음 오류] {error}")

    def _sound_test_windows(self) -> None:
        try:
            import winsound

            winsound.Beep(SOUND_TEST_LOW_FREQUENCY, SOUND_TEST_DURATION_MS)
            time.sleep(0.08)
            winsound.Beep(SOUND_TEST_HIGH_FREQUENCY, SOUND_TEST_DURATION_MS)
            print("[사운드 테스트] Windows 경고음 요청 전송 완료")
        except Exception as error:
            self.last_sound_error = str(error)
            print(f"[사운드 테스트 실패] {error}")

    @classmethod
    def _pick_primary_reason(cls, reasons: list[str]) -> Optional[str]:
        for reason in cls.PRIORITY:
            if reason in reasons:
                return reason
        return reasons[0] if reasons else None
