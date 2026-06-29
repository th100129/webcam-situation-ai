from __future__ import annotations

import time
from typing import Dict, List


class MetricsTracker:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.frame_count = 0
        self.fps_samples: List[float] = []
        self.missions_started = 0
        self.missions_completed = 0
        self.mission_timeouts = 0
        self.reaction_times: List[float] = []

    def update_frame(self, fps: float) -> None:
        self.frame_count += 1
        self.fps_samples.append(fps)
        if len(self.fps_samples) > 300:
            self.fps_samples.pop(0)

    def mission_started(self) -> None:
        self.missions_started += 1

    def mission_completed(self, reaction_time: float) -> None:
        self.missions_completed += 1
        self.reaction_times.append(reaction_time)

    def mission_timed_out(self) -> None:
        self.mission_timeouts += 1

    def summary(self) -> Dict:
        elapsed = time.monotonic() - self.started_at
        success_rate = (self.missions_completed / self.missions_started * 100.0) if self.missions_started else 0.0
        average_reaction = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0.0
        average_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
        return {
            "session_seconds": round(elapsed, 2),
            "frames": self.frame_count,
            "average_fps": round(average_fps, 2),
            "missions_started": self.missions_started,
            "missions_completed": self.missions_completed,
            "mission_timeouts": self.mission_timeouts,
            "mission_success_rate": round(success_rate, 2),
            "average_reaction_seconds": round(average_reaction, 2),
        }
