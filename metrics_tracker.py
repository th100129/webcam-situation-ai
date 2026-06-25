# metrics_tracker.py

import time
from typing import Dict, Optional


class MetricsTracker:
    """
    FPS, 미션 성공률, 반응 시간, 오탐/미탐 횟수를 관리하는 클래스.
    """

    def __init__(self):
        self.prev_frame_time = time.time()
        self.fps_values = []

        self.focus_alert_count = 0

        self.mission_attempt_count = 0
        self.mission_success_count = 0
        self.mission_fail_count = 0

        self.reaction_times = []

        self.false_positive_count = 0
        self.false_negative_count = 0

        self.current_mission_start_time: Optional[float] = None

    def update_fps(self) -> float:
        """
        프레임마다 호출해서 FPS를 계산한다.
        """
        now = time.time()
        elapsed = now - self.prev_frame_time
        self.prev_frame_time = now

        if elapsed <= 0:
            return 0.0

        fps = 1.0 / elapsed
        self.fps_values.append(fps)

        return fps

    def record_focus_alert(self):
        self.focus_alert_count += 1

    def start_mission_timer(self):
        self.current_mission_start_time = time.time()
        self.mission_attempt_count += 1

    def get_current_reaction_time(self) -> Optional[float]:
        if self.current_mission_start_time is None:
            return None

        return time.time() - self.current_mission_start_time

    def record_mission_success(self) -> Optional[float]:
        """
        미션 성공 기록.
        성공까지 걸린 반응 시간을 반환한다.
        """
        reaction_time = self.get_current_reaction_time()

        self.mission_success_count += 1

        if reaction_time is not None:
            self.reaction_times.append(reaction_time)

        self.current_mission_start_time = None

        return reaction_time

    def record_mission_fail(self) -> Optional[float]:
        """
        미션 실패 기록.
        실패까지 걸린 시간을 반환한다.
        """
        reaction_time = self.get_current_reaction_time()

        self.mission_fail_count += 1

        if reaction_time is not None:
            self.reaction_times.append(reaction_time)

        self.current_mission_start_time = None

        return reaction_time

    def record_false_positive(self):
        """
        오탐:
        실제로 집중 흐림이 아닌데 시스템이 집중 흐림으로 감지한 경우.
        수동 테스트하면서 키 입력이나 별도 조건으로 기록하면 된다.
        """
        self.false_positive_count += 1

    def record_false_negative(self):
        """
        미탐:
        실제로 집중 흐림인데 시스템이 감지하지 못한 경우.
        수동 테스트하면서 키 입력이나 별도 조건으로 기록하면 된다.
        """
        self.false_negative_count += 1

    def get_avg_fps(self) -> float:
        if not self.fps_values:
            return 0.0

        return sum(self.fps_values) / len(self.fps_values)

    def get_avg_reaction_time(self) -> float:
        if not self.reaction_times:
            return 0.0

        return sum(self.reaction_times) / len(self.reaction_times)

    def get_mission_success_rate(self) -> float:
        if self.mission_attempt_count == 0:
            return 0.0

        return self.mission_success_count / self.mission_attempt_count * 100

    def get_error_rates(self) -> Dict[str, float]:
        total = self.false_positive_count + self.false_negative_count

        if total == 0:
            return {
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            }

        return {
            "false_positive_rate": self.false_positive_count / total * 100,
            "false_negative_rate": self.false_negative_count / total * 100,
        }

    def get_summary(self, total_events: int = 0) -> Dict:
        error_rates = self.get_error_rates()

        return {
            "total_events": total_events,
            "focus_alert_count": self.focus_alert_count,
            "mission_attempt_count": self.mission_attempt_count,
            "mission_success_count": self.mission_success_count,
            "mission_fail_count": self.mission_fail_count,
            "mission_success_rate": self.get_mission_success_rate(),
            "avg_reaction_time": self.get_avg_reaction_time(),
            "avg_fps": self.get_avg_fps(),
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "false_positive_rate": error_rates["false_positive_rate"],
            "false_negative_rate": error_rates["false_negative_rate"],
        }