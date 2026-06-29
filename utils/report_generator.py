from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class ReportGenerator:
    @staticmethod
    def save(report_dir: Path, summary: Dict) -> Path:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return path
