# session_logger.py

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import LOG_DIR


class SessionLogger:
    """
    실행 중 발생한 이벤트를 CSV와 JSON으로 저장하는 클래스.
    """

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"session_{timestamp}.csv"
        self.json_path = self.log_dir / f"session_{timestamp}.json"

        self.events: List[Dict] = []

        self.csv_columns = [
            "timestamp",
            "event_type",
            "focus_state",
            "reason",
            "mission_type",
            "mission_success",
            "reaction_time",
            "fps",
            "message",
            "raw_summary",
            "stable_summary",
            "hand_summary",
        ]

        self._init_csv()

    def log_event(
        self,
        event_type: str,
        focus_state: str = "",
        reason: str = "",
        mission_type: str = "",
        mission_success: Optional[bool] = None,
        reaction_time: Optional[float] = None,
        fps: Optional[float] = None,
        message: str = "",
        raw_summary: str = "",
        stable_summary: str = "",
        hand_summary: str = "",
    ) -> Dict:
        """
        이벤트 1개를 저장한다.
        """
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "focus_state": focus_state,
            "reason": reason,
            "mission_type": mission_type,
            "mission_success": mission_success,
            "reaction_time": self._round_or_none(reaction_time),
            "fps": self._round_or_none(fps),
            "message": message,
            "raw_summary": raw_summary,
            "stable_summary": stable_summary,
            "hand_summary": hand_summary,
        }

        self.events.append(event)
        self._append_csv(event)
        self._save_json()

        return event

    def get_events(self) -> List[Dict]:
        return self.events

    def get_log_paths(self) -> Dict[str, str]:
        return {
            "csv_path": str(self.csv_path),
            "json_path": str(self.json_path),
        }

    def _init_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_columns)
            writer.writeheader()

    def _append_csv(self, event: Dict):
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_columns)
            writer.writerow(event)

    def _save_json(self):
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(
                self.events,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def _round_or_none(self, value):
        if value is None:
            return None

        try:
            return round(float(value), 3)
        except TypeError:
            return None