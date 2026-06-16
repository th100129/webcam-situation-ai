import os
import random
import threading
import time

import cv2


class RepeatingBeepAlarm:
    def __init__(
        self,
        frequency=1200,
        duration_ms=180,
        interval_seconds=0.8,
    ):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.interval_seconds = interval_seconds

        self.running = False
        self.thread = None

    def start(self):
        if os.name != "nt":
            self.running = True
            return

        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        try:
            import winsound
        except Exception:
            return

        while self.running:
            try:
                winsound.Beep(self.frequency, self.duration_ms)
            except Exception:
                pass

            time.sleep(self.interval_seconds)


class MissionManager:
    def __init__(
        self,
        mission_hold_seconds=1.0,
        mission_cooldown_seconds=2.0,
        enable_alarm=True,
    ):
        self.mission_hold_seconds = mission_hold_seconds
        self.mission_cooldown_seconds = mission_cooldown_seconds
        self.enable_alarm = enable_alarm

        self.alarm = RepeatingBeepAlarm()

        self.current_mission = None
        self.mission_started_at = None
        self.success_started_at = None
        self.completed_at = None

        self.completed_for_current_episode = False

        # OpenCV putText는 한글 출력이 깨지므로 화면 표시 문구는 영어로 사용
        self.missions = [
            {
                "id": "open_palm",
                "type": "hand_gesture",
                "target": "open_palm",
                "title": "Open Palm",
                "description": "Show your open palm for 1 second.",
            },
            {
                "id": "fist",
                "type": "hand_gesture",
                "target": "fist",
                "title": "Make a Fist",
                "description": "Make a fist and hold it for 1 second.",
            },
            {
                "id": "pointing",
                "type": "hand_gesture",
                "target": "pointing",
                "title": "Pointing",
                "description": "Point with your index finger for 1 second.",
            },
            {
                "id": "two_fingers",
                "type": "hand_gesture",
                "target": "two_fingers",
                "title": "Two Fingers",
                "description": "Show two fingers for 1 second.",
            },
            {
                "id": "left_hand_up",
                "type": "pose",
                "target": "left_hand_up",
                "title": "Raise Left Hand",
                "description": "Raise your left hand above your shoulder.",
            },
            {
                "id": "right_hand_up",
                "type": "pose",
                "target": "right_hand_up",
                "title": "Raise Right Hand",
                "description": "Raise your right hand above your shoulder.",
            },
            {
                "id": "both_hands_up",
                "type": "pose",
                "target": "both_hands_up",
                "title": "Raise Both Hands",
                "description": "Raise both hands above your shoulders.",
            },
        ]

    def update(self, focus_action, hand_infos, pose_infos, frame_shape):
        now = time.time()

        result = {
            "active": False,
            "completed": False,
            "started": False,
            "mission_id": None,
            "mission_title": "none",
            "mission_description": "none",
            "progress": 0.0,
            "success_detected": False,
            "message": "no mission",
            "alarm_running": self.alarm.running,
            "overlay_lines": [],
        }

        focus_active = focus_action["active"]

        if focus_active and not self.current_mission and not self.completed_for_current_episode:
            self._start_random_mission(now)
            result["started"] = True

        if self.current_mission:
            success_detected = self._check_mission_success(
                mission=self.current_mission,
                hand_infos=hand_infos,
                pose_infos=pose_infos,
                frame_shape=frame_shape,
            )

            if success_detected:
                if self.success_started_at is None:
                    self.success_started_at = now

                progress = min(
                    (now - self.success_started_at) / self.mission_hold_seconds,
                    1.0,
                )

                if progress >= 1.0:
                    self._complete_mission(now)

                    result["active"] = False
                    result["completed"] = True
                    result["mission_id"] = self.current_mission_id_after_complete
                    result["mission_title"] = self.current_mission_title_after_complete
                    result["mission_description"] = "Mission complete"
                    result["progress"] = 1.0
                    result["success_detected"] = True
                    result["message"] = "mission completed"
                    result["alarm_running"] = self.alarm.running
                    result["overlay_lines"] = [
                        "MISSION COMPLETE",
                        "Alarm stopped.",
                    ]

                    return result
            else:
                self.success_started_at = None
                progress = 0.0

            result["active"] = True
            result["mission_id"] = self.current_mission["id"]
            result["mission_title"] = self.current_mission["title"]
            result["mission_description"] = self.current_mission["description"]
            result["progress"] = progress
            result["success_detected"] = success_detected
            result["message"] = "mission active"
            result["alarm_running"] = self.alarm.running
            result["overlay_lines"] = [
                "MISSION REQUIRED",
                self.current_mission["title"],
                self.current_mission["description"],
                f"Progress: {int(progress * 100)}%",
            ]

            return result

        if not focus_active:
            self.completed_for_current_episode = False

        return result

    def stop(self):
        self.alarm.stop()

    def make_signature(self, mission_result):
        return (
            mission_result["active"],
            mission_result["completed"],
            mission_result["mission_id"],
            int(mission_result["progress"] * 10),
        )

    def _start_random_mission(self, now):
        self.current_mission = random.choice(self.missions)
        self.mission_started_at = now
        self.success_started_at = None
        self.completed_at = None

        if self.enable_alarm:
            self.alarm.start()

    def _complete_mission(self, now):
        self.current_mission_id_after_complete = self.current_mission["id"]
        self.current_mission_title_after_complete = self.current_mission["title"]

        self.completed_at = now
        self.completed_for_current_episode = True

        self.current_mission = None
        self.mission_started_at = None
        self.success_started_at = None

        self.alarm.stop()

    def _check_mission_success(self, mission, hand_infos, pose_infos, frame_shape):
        mission_type = mission["type"]
        target = mission["target"]

        if mission_type == "hand_gesture":
            return self._check_hand_gesture(
                hand_infos=hand_infos,
                target=target,
            )

        if mission_type == "pose":
            return self._check_pose(
                pose_infos=pose_infos,
                target=target,
                frame_shape=frame_shape,
            )

        return False

    def _check_hand_gesture(self, hand_infos, target):
        for hand in hand_infos:
            gesture = self._get_value(hand, ["gesture"], None)

            if gesture == target:
                return True

        return False

    def _check_pose(self, pose_infos, target, frame_shape):
        if not pose_infos:
            return False

        pose = pose_infos[0]
        landmarks = self._get_value(pose, ["landmarks"], None)

        if not landmarks:
            return False

        height, width = frame_shape[:2]
        points = self._convert_landmarks_to_points(landmarks, frame_shape)

        left_shoulder = self._get_point(points, 11)
        right_shoulder = self._get_point(points, 12)
        left_wrist = self._get_point(points, 15)
        right_wrist = self._get_point(points, 16)

        margin = int(height * 0.04)

        left_hand_up = (
            left_shoulder is not None
            and left_wrist is not None
            and left_wrist[1] < left_shoulder[1] - margin
        )

        right_hand_up = (
            right_shoulder is not None
            and right_wrist is not None
            and right_wrist[1] < right_shoulder[1] - margin
        )

        if target == "left_hand_up":
            return left_hand_up

        if target == "right_hand_up":
            return right_hand_up

        if target == "both_hands_up":
            return left_hand_up and right_hand_up

        return False

    def _get_point(self, points, index):
        if index >= len(points):
            return None

        point = points[index]

        if len(point) >= 4:
            visibility = point[3]

            if visibility is not None and visibility < 0.3:
                return None

        return point

    def _convert_landmarks_to_points(self, raw_landmarks, frame_shape):
        height, width = frame_shape[:2]

        points = []

        for landmark in raw_landmarks:
            x = None
            y = None
            z = 0.0
            visibility = 1.0

            if isinstance(landmark, dict):
                x = landmark.get("x")
                y = landmark.get("y")
                z = landmark.get("z", 0.0)
                visibility = landmark.get("visibility", 1.0)

            elif isinstance(landmark, (list, tuple)):
                if len(landmark) >= 2:
                    x = landmark[0]
                    y = landmark[1]

                if len(landmark) >= 3:
                    z = landmark[2]

                if len(landmark) >= 4:
                    visibility = landmark[3]

            else:
                x = getattr(landmark, "x", None)
                y = getattr(landmark, "y", None)
                z = getattr(landmark, "z", 0.0)
                visibility = getattr(landmark, "visibility", 1.0)

            if x is None or y is None:
                continue

            if visibility is None:
                visibility = 1.0

            if 0 <= x <= 2 and 0 <= y <= 2:
                px = int(x * width)
                py = int(y * height)
            else:
                px = int(x)
                py = int(y)

            points.append((px, py, z, visibility))

        return points

    def _get_value(self, item, names, default=None):
        for name in names:
            if isinstance(item, dict) and name in item:
                return item[name]

            if hasattr(item, name):
                return getattr(item, name)

        return default


def draw_mission_overlay(frame, mission_result):
    if not mission_result["overlay_lines"]:
        return frame

    overlay = frame.copy()
    height, width = frame.shape[:2]

    box_width = min(700, width - 40)
    box_height = 170

    x1 = int((width - box_width) / 2)
    y1 = height - box_height - 25
    x2 = x1 + box_width
    y2 = y1 + box_height

    if mission_result["completed"]:
        color = (0, 180, 0)
    else:
        color = (0, 0, 180)

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1,
    )

    alpha = 0.78
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
    line_gap = 28

    for i, line in enumerate(mission_result["overlay_lines"]):
        font_scale = 0.78 if i == 0 else 0.62
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