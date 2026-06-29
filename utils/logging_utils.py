from __future__ import annotations

from typing import Dict


def print_change_log(focus_analysis: Dict, mission_result: Dict, event_text: str) -> None:
    print("\n" + "=" * 70)
    print(f"[EVENT] {event_text}")
    print(
        "[FOCUS] "
        f"score={focus_analysis['score']} level={focus_analysis['level']} "
        f"reasons={focus_analysis['active_reasons']} "
        f"face={focus_analysis['face']['direction']} eyes={focus_analysis['face']['eye_state']}"
    )
    if mission_result["active"]:
        print(f"[MISSION] {mission_result['mission_title']} | progress={mission_result['progress'] * 100:.0f}%")
    elif mission_result["completed"]:
        print(f"[MISSION] completed: {mission_result['mission_title']}")
    print("=" * 70)
