# report_generator.py

from datetime import datetime
from pathlib import Path
from typing import Dict

from config import REPORT_DIR


class ReportGenerator:
    """
    세션 종료 시 요약 리포트를 txt 파일로 저장하는 클래스.
    """

    def __init__(self, report_dir: Path = REPORT_DIR):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_report(
        self,
        session_data: Dict,
        summary_text: str,
        csv_path: str = "",
        json_path: str = "",
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"report_{timestamp}.txt"

        report_text = self._build_report_text(
            session_data=session_data,
            summary_text=summary_text,
            csv_path=csv_path,
            json_path=json_path,
        )

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report_text)

        return str(report_path)

    def _build_report_text(
        self,
        session_data: Dict,
        summary_text: str,
        csv_path: str,
        json_path: str,
    ) -> str:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""
[Focus Trigger 세션 요약 리포트]

생성 시간: {created_at}

1. 세션 요약
{summary_text}

2. 정량 지표
- 전체 이벤트 수: {session_data.get("total_events", 0)}
- 집중 흐림 감지 횟수: {session_data.get("focus_alert_count", 0)}
- 미션 시도 횟수: {session_data.get("mission_attempt_count", 0)}
- 미션 성공 횟수: {session_data.get("mission_success_count", 0)}
- 미션 실패 횟수: {session_data.get("mission_fail_count", 0)}
- 미션 성공률: {session_data.get("mission_success_rate", 0):.1f}%
- 평균 반응 시간: {session_data.get("avg_reaction_time", 0):.2f}초
- 평균 FPS: {session_data.get("avg_fps", 0):.1f}
- 오탐 횟수: {session_data.get("false_positive_count", 0)}
- 미탐 횟수: {session_data.get("false_negative_count", 0)}
- 오탐률: {session_data.get("false_positive_rate", 0):.1f}%
- 미탐률: {session_data.get("false_negative_rate", 0):.1f}%

3. 저장된 로그 파일
- CSV 로그: {csv_path}
- JSON 로그: {json_path}
""".strip()