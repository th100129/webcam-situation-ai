import time

import cv2


class FocusActionManager:
    def __init__(
        self,
        enter_hold_seconds=0.8,
        recover_hold_seconds=2.0,
        action_cooldown_seconds=6.0,
        screenshot_after_seconds=8.0,
        warning_score_threshold=50,
        high_score_threshold=75,
        enable_auto_screenshot=True,
    ):
        self.enter_hold_seconds = enter_hold_seconds
        self.recover_hold_seconds = recover_hold_seconds
        self.action_cooldown_seconds = action_cooldown_seconds
        self.screenshot_after_seconds = screenshot_after_seconds
        self.warning_score_threshold = warning_score_threshold
        self.high_score_threshold = high_score_threshold
        self.enable_auto_screenshot = enable_auto_screenshot

        self.state = "normal"

        self.focus_blur_since = None
        self.recover_since = None

        self.last_action_time = 0.0
        self.last_action_level = None

        self.screenshot_saved_in_episode = False

    def update(self, focus_analysis):
        now = time.time()

        focus_blur_detected = self._is_focus_blur_detected(focus_analysis)
        action_level = self._get_action_level(focus_analysis)
        message = self._make_message(focus_analysis, action_level)

        result = {
            "state": self.state,
            "focus_blur_detected": focus_blur_detected,
            "active": False,
            "triggered": False,
            "recovered": False,
            "action_level": action_level,
            "message": message,
            "duration": 0.0,
            "should_save_screenshot": False,
            "overlay_lines": [],
        }

        if focus_blur_detected:
            self.recover_since = None

            if self.focus_blur_since is None:
                self.focus_blur_since = now
                self.screenshot_saved_in_episode = False

            duration = now - self.focus_blur_since
            result["duration"] = duration

            if duration < self.enter_hold_seconds:
                self.state = "watching"
                result["state"] = self.state
                result["overlay_lines"] = [
                    "Focus check...",
                    "Possible focus blur detected",
                ]
                return result

            self.state = "active"
            result["state"] = self.state
            result["active"] = True

            should_trigger_action = self._should_trigger_action(
                now=now,
                action_level=action_level,
            )

            if should_trigger_action:
                result["triggered"] = True
                self.last_action_time = now
                self.last_action_level = action_level

            if (
                self.enable_auto_screenshot
                and not self.screenshot_saved_in_episode
                and duration >= self.screenshot_after_seconds
            ):
                result["should_save_screenshot"] = True
                self.screenshot_saved_in_episode = True

            result["overlay_lines"] = self._make_overlay_lines(
                focus_analysis=focus_analysis,
                action_level=action_level,
                duration=duration,
            )

            return result

        self.focus_blur_since = None
        self.screenshot_saved_in_episode = False

        if self.state in ["watching", "active"]:
            if self.recover_since is None:
                self.recover_since = now

            recover_duration = now - self.recover_since

            if recover_duration >= self.recover_hold_seconds:
                self.state = "normal"
                self.recover_since = None
                self.last_action_level = None

                result["state"] = self.state
                result["recovered"] = True
                result["message"] = "Focus recovered"
                result["overlay_lines"] = [
                    "Focus recovered",
                    "Back to normal",
                ]

                return result

            self.state = "recovering"
            result["state"] = self.state
            result["overlay_lines"] = [
                "Recovering focus...",
                f"{recover_duration:.1f}s",
            ]

            return result

        self.state = "normal"
        result["state"] = self.state
        result["message"] = "Focus normal"

        return result

    def make_signature(self, action_result):
        return (
            action_result["state"],
            action_result["active"],
            action_result["triggered"],
            action_result["recovered"],
            action_result["action_level"],
        )

    def _is_focus_blur_detected(self, focus_analysis):
        score = focus_analysis["score"]
        stable = focus_analysis["stable_indicators"]

        strong_single_signal = (
            stable.get("face_missing", False)
            or stable.get("face_away", False)
            or stable.get("head_down", False)
            or stable.get("eyes_closed", False)
        )

        combined_signal = (
            score >= self.warning_score_threshold
            or (
                stable.get("hand_face_contact", False)
                and stable.get("pose_unstable", False)
            )
            or (
                stable.get("hand_face_contact", False)
                and stable.get("head_down", False)
            )
        )

        return strong_single_signal or combined_signal

    def _get_action_level(self, focus_analysis):
        score = focus_analysis["score"]
        stable = focus_analysis["stable_indicators"]

        if score >= self.high_score_threshold:
            return "high"

        if stable.get("face_missing", False):
            return "high"

        if stable.get("eyes_closed", False):
            return "high"

        if score >= self.warning_score_threshold:
            return "warning"

        if stable.get("head_down", False):
            return "warning"

        if stable.get("face_away", False):
            return "warning"

        return "caution"

    def _should_trigger_action(self, now, action_level):
        cooldown_passed = (
            now - self.last_action_time
        ) >= self.action_cooldown_seconds

        level_changed_to_high = (
            action_level == "high"
            and self.last_action_level != "high"
        )

        return cooldown_passed or level_changed_to_high

    def _make_message(self, focus_analysis, action_level):
        stable = focus_analysis["stable_indicators"]

        if stable.get("face_missing", False):
            return "Face is missing. Please return to the camera."

        if stable.get("eyes_closed", False):
            return "Eyes closed detected. Please complete the mission."

        if stable.get("head_down", False):
            return "Head down detected. Please look forward."

        if stable.get("face_away", False):
            return "Face direction is away. Please look at the screen."

        if stable.get("hand_face_contact", False):
            return "Hand-face contact detected. Please check your posture."

        if stable.get("pose_unstable", False):
            return "Unstable posture detected. Please sit steadily."

        if action_level == "high":
            return "High focus blur possibility detected."

        if action_level == "warning":
            return "Focus warning detected."

        return "Focus caution detected."

    def _make_overlay_lines(self, focus_analysis, action_level, duration):
        score = focus_analysis["score"]
        active_reasons = focus_analysis["active_reasons"]

        if active_reasons:
            reason_text = ", ".join(active_reasons)
        else:
            reason_text = "unknown"

        if action_level == "high":
            title = "HIGH FOCUS BLUR"
        elif action_level == "warning":
            title = "FOCUS WARNING"
        else:
            title = "FOCUS CAUTION"

        return [
            title,
            f"Score: {score}",
            f"Reason: {reason_text}",
            f"Duration: {duration:.1f}s",
            self._make_short_korean_message(active_reasons),
        ]

    def _make_short_korean_message(self, active_reasons):
        if "face_missing" in active_reasons:
            return "얼굴이 화면에서 벗어났습니다."

        if "eyes_closed" in active_reasons:
            return "눈 감김이 5초 이상 감지되었습니다."

        if "head_down" in active_reasons:
            return "고개 숙임이 감지되었습니다."

        if "face_away" in active_reasons:
            return "시선/얼굴 방향이 벗어났습니다."

        if "hand_face_contact" in active_reasons:
            return "손이 얼굴 근처에 있습니다."

        if "pose_unstable" in active_reasons:
            return "자세 흔들림이 감지되었습니다."

        return "집중 흐림 가능성이 있습니다."


def draw_focus_action_overlay(frame, action_result):
    if not action_result["overlay_lines"]:
        return frame

    if action_result["state"] == "normal":
        return frame

    overlay = frame.copy()

    height, width = frame.shape[:2]

    box_width = min(620, width - 40)
    box_height = 150

    x1 = int((width - box_width) / 2)
    y1 = 20
    x2 = x1 + box_width
    y2 = y1 + box_height

    action_level = action_result["action_level"]

    if action_result["recovered"]:
        color = (0, 180, 0)
    elif action_level == "high":
        color = (0, 0, 255)
    elif action_level == "warning":
        color = (0, 165, 255)
    else:
        color = (0, 255, 255)

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1,
    )

    alpha = 0.72
    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    text_x = x1 + 20
    text_y = y1 + 35
    line_gap = 26

    for i, line in enumerate(action_result["overlay_lines"]):
        font_scale = 0.75 if i == 0 else 0.62
        thickness = 2 if i == 0 else 1

        cv2.putText(
            frame,
            line,
            (text_x, text_y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return frame