from __future__ import annotations

import os
from typing import Dict

from config import LLM_MODEL_NAME, USE_LLM_API
from feedback.prompt_templates import MISSION_PROMPT, RETRY_PROMPT, SUCCESS_PROMPT
from focus.focus_analyzer import FocusAnalyzer


class LLMManager:
    """기본은 API 없이 동작하는 문구 생성기. USE_LLM_API=True일 때만 OpenAI API를 시도한다."""

    def __init__(self) -> None:
        self.enabled = USE_LLM_API and bool(os.getenv("OPENAI_API_KEY"))
        self.client = None
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception:
                self.enabled = False

    def mission_message(self, mission_result: Dict) -> str:
        reason_label = FocusAnalyzer.REASON_LABELS.get(mission_result.get("reason"), "집중 흐림")
        fallback = f"{reason_label}이 감지됐어요. {mission_result['instruction']}"
        if not self.enabled:
            return fallback
        prompt = MISSION_PROMPT.format(
            reason=reason_label,
            mission_title=mission_result["mission_title"],
            instruction=mission_result["instruction"],
        )
        return self._ask(prompt, fallback)

    def success_message(self, mission_result: Dict) -> str:
        reaction = mission_result.get("reaction_time") or 0.0
        fallback = f"좋아요! {reaction:.1f}초 만에 미션을 완료했어요. 다시 집중해볼까요?"
        if not self.enabled:
            return fallback
        prompt = SUCCESS_PROMPT.format(mission_title=mission_result["mission_title"], reaction_time=reaction)
        return self._ask(prompt, fallback)

    def retry_message(self, mission_result: Dict) -> str:
        fallback = f"아직 경고가 유지돼요. {mission_result['instruction']}"
        if not self.enabled:
            return fallback
        prompt = RETRY_PROMPT.format(mission_title=mission_result["mission_title"])
        return self._ask(prompt, fallback)

    def _ask(self, prompt: str, fallback: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=80,
            )
            return response.choices[0].message.content.strip() or fallback
        except Exception:
            return fallback
