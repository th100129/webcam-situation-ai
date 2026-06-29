from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class SessionLogger:
    """이벤트를 JSONL로 저장한다. 각 줄이 하나의 독립적인 JSON 이벤트다."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def log(self, event_type: str, payload: Dict) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
