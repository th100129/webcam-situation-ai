from __future__ import annotations

import time
from collections import Counter
from typing import Dict, Iterable, List


class ObjectStateTracker:
    """객체가 잠깐 보인 것과 일정 시간 유지된 것을 구분한다."""

    def __init__(self, appear_threshold: float, disappear_threshold: float) -> None:
        self.appear_threshold = appear_threshold
        self.disappear_threshold = disappear_threshold
        self.states: Dict[str, Dict] = {}

    def update(self, detections: Iterable[Dict]) -> None:
        now = time.monotonic()
        detections = list(detections)
        counts = Counter(det["name"] for det in detections)
        confidence_by_name: Dict[str, float] = {}
        for det in detections:
            confidence_by_name[det["name"]] = max(confidence_by_name.get(det["name"], 0.0), det["confidence"])

        for name, count in counts.items():
            if name not in self.states:
                self.states[name] = {
                    "first_seen": now,
                    "last_seen": now,
                    "stable": False,
                    "count": count,
                    "max_confidence": confidence_by_name[name],
                }
            else:
                state = self.states[name]
                state["last_seen"] = now
                state["count"] = count
                state["max_confidence"] = confidence_by_name[name]

            if now - self.states[name]["first_seen"] >= self.appear_threshold:
                self.states[name]["stable"] = True

        expired = [
            name
            for name, state in self.states.items()
            if now - state["last_seen"] >= self.disappear_threshold
        ]
        for name in expired:
            del self.states[name]

    def get_stable_objects(self) -> List[Dict]:
        now = time.monotonic()
        result = []
        for name, state in self.states.items():
            if not state["stable"]:
                continue
            result.append(
                {
                    "name": name,
                    "count": state["count"],
                    "duration": now - state["first_seen"],
                    "max_confidence": state["max_confidence"],
                }
            )
        return sorted(result, key=lambda item: item["name"])

    @staticmethod
    def summary(stable_objects: Iterable[Dict]) -> str:
        objects = list(stable_objects)
        if not objects:
            return "none"
        return ", ".join(f"{obj['name']}:{obj['count']} ({obj['duration']:.1f}s)" for obj in objects)
