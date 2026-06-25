# llm_manager.py

import os
import random
from typing import Dict, Optional

from config import (
    LLM_MODEL_NAME,
    MISSION_DESCRIPTIONS,
    MISSION_LABELS,
    REASON_LABELS,
    USE_LLM_API,
)
from feedback.prompt_templates import (
    FAIL_PROMPT_TEMPLATE,
    MISSION_PROMPT_TEMPLATE,
    SESSION_SUMMARY_PROMPT_TEMPLATE,
    SUCCESS_PROMPT_TEMPLATE,
)


class LLMManager:
    """
    CV 분석 결과를 받아서 사용자에게 보여줄 문구를 생성하는 클래스.

    현재 기본값은 템플릿 기반 생성이다.
    나중에 실제 LLM API를 붙이고 싶으면 USE_LLM_API=True로 바꾸면 된다.
    """

    def __init__(self, use_llm_api: bool = USE_LLM_API):
        self.use_llm_api = use_llm_api

    def build_cv_input(
        self,
        focus_state: str,
        reason: str,
        mission_type: Optional[str] = None,
        reason_detail: str = "",
        extra: Optional[Dict] = None,
    ) -> Dict:
        """
        CV 분석 결과를 LLM 입력용 딕셔너리로 변환한다.
        """
        mission_type = mission_type or "none"

        return {
            "focus_state": focus_state,
            "reason": reason,
            "reason_label": REASON_LABELS.get(reason, reason),
            "reason_detail": reason_detail or "상세 정보 없음",
            "mission_type": mission_type,
            "mission_label": MISSION_LABELS.get(mission_type, mission_type),
            "mission_description": MISSION_DESCRIPTIONS.get(
                mission_type,
                "주어진 미션을 수행해주세요.",
            ),
            "extra": extra or {},
        }

    def generate_mission_message(self, cv_input: Dict) -> str:
        """
        집중 흐림 감지 후 미션 안내 문구 생성
        """
        prompt = MISSION_PROMPT_TEMPLATE.format(
            focus_state=cv_input.get("focus_state", "unknown"),
            reason_label=cv_input.get("reason_label", "알 수 없음"),
            reason_detail=cv_input.get("reason_detail", "상세 정보 없음"),
            mission_label=cv_input.get("mission_label", "미션"),
            mission_description=cv_input.get(
                "mission_description",
                "주어진 미션을 수행해주세요.",
            ),
        )

        if self.use_llm_api:
            return self._call_llm(prompt)

        return self._generate_mission_message_by_template(cv_input)

    def generate_success_feedback(
        self,
        mission_type: str,
        reaction_time: Optional[float] = None,
    ) -> str:
        """
        미션 성공 피드백 생성
        """
        mission_label = MISSION_LABELS.get(mission_type, mission_type)

        prompt = SUCCESS_PROMPT_TEMPLATE.format(
            mission_label=mission_label,
            reaction_time=self._format_seconds(reaction_time),
        )

        if self.use_llm_api:
            return self._call_llm(prompt)

        messages = [
            f"좋아요! {mission_label} 미션 성공이에요. 다시 집중해볼까요?",
            f"{mission_label} 성공! 알림을 멈추고 다시 이어가면 돼요.",
            f"잘했어요. {mission_label} 동작이 확인됐어요!",
            f"미션 완료! 잠깐 흐트러졌지만 바로 돌아왔네요.",
        ]
        return random.choice(messages)

    def generate_fail_feedback(
        self,
        mission_type: str,
        timeout_seconds: float,
    ) -> str:
        """
        미션 실패 또는 시간 초과 피드백 생성
        """
        mission_label = MISSION_LABELS.get(mission_type, mission_type)

        prompt = FAIL_PROMPT_TEMPLATE.format(
            mission_label=mission_label,
            timeout_seconds=timeout_seconds,
        )

        if self.use_llm_api:
            return self._call_llm(prompt)

        messages = [
            f"{mission_label} 미션이 아직 인식되지 않았어요. 손이 카메라에 잘 보이게 다시 해볼까요?",
            f"조금만 더 또렷하게 보여주세요. {mission_label} 동작을 다시 시도해봐요.",
            "미션 인식 시간이 초과됐어요. 카메라 위치를 확인하고 다시 시도해보세요.",
            "아직 성공으로 판단되지 않았어요. 손동작을 화면 중앙에서 보여주세요.",
        ]
        return random.choice(messages)

    def generate_session_summary(self, session_data: Dict) -> str:
        """
        세션 종료 후 요약 문구 생성
        """
        prompt = SESSION_SUMMARY_PROMPT_TEMPLATE.format(
            total_events=session_data.get("total_events", 0),
            focus_alert_count=session_data.get("focus_alert_count", 0),
            mission_attempt_count=session_data.get("mission_attempt_count", 0),
            mission_success_count=session_data.get("mission_success_count", 0),
            mission_fail_count=session_data.get("mission_fail_count", 0),
            mission_success_rate=round(
                session_data.get("mission_success_rate", 0),
                1,
            ),
            avg_reaction_time=round(
                session_data.get("avg_reaction_time", 0),
                2,
            ),
            avg_fps=round(
                session_data.get("avg_fps", 0),
                1,
            ),
            false_positive_count=session_data.get("false_positive_count", 0),
            false_negative_count=session_data.get("false_negative_count", 0),
        )

        if self.use_llm_api:
            return self._call_llm(prompt)

        focus_alert_count = session_data.get("focus_alert_count", 0)
        success_count = session_data.get("mission_success_count", 0)
        fail_count = session_data.get("mission_fail_count", 0)
        success_rate = session_data.get("mission_success_rate", 0)
        avg_fps = session_data.get("avg_fps", 0)

        return (
            f"이번 세션에서는 집중 흐림이 총 {focus_alert_count}회 감지되었습니다. "
            f"미션은 {success_count}회 성공, {fail_count}회 실패로 기록되었고 "
            f"성공률은 {success_rate:.1f}%입니다. "
            f"평균 FPS는 {avg_fps:.1f}로 측정되어 실시간 처리 상태를 확인할 수 있었습니다."
        )

    def _generate_mission_message_by_template(self, cv_input: Dict) -> str:
        reason = cv_input.get("reason", "unknown")
        mission_description = cv_input.get(
            "mission_description",
            "주어진 미션을 수행해주세요.",
        )

        reason_messages = {
            "eyes_closed": "눈을 오래 감고 있었어요.",
            "head_down": "고개가 아래로 향한 것 같아요.",
            "face_missing": "얼굴이 화면에서 잠시 사라졌어요.",
            "hand_face_contact": "손이 얼굴 근처에 오래 머문 것 같아요.",
            "unstable_pose": "자세가 조금 흔들린 것 같아요.",
            "object_or_hand_change": "상태 변화가 감지됐어요.",
            "unknown": "집중이 잠깐 흐려진 것 같아요.",
        }

        reason_text = reason_messages.get(
            reason,
            "집중이 잠깐 흐려진 것 같아요.",
        )

        return f"{reason_text} {mission_description}"

    def _call_llm(self, prompt: str) -> str:
        """
        실제 LLM API 호출부.

        openai 패키지를 설치하고 OPENAI_API_KEY를 환경변수로 설정하면 사용 가능.
        실패하면 템플릿 문구로 대체한다.
        """
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return "LLM API 키가 없어 기본 문구로 안내할게요. 화면 중앙에서 미션을 수행해주세요."

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "너는 짧고 친근한 한국어 피드백 문구를 작성하는 도우미야.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=120,
            )

            return response.choices[0].message.content.strip()

        except Exception as error:
            print(f"[LLM 호출 실패] {error}")
            return "문구 생성에 실패했지만 괜찮아요. 화면 중앙에서 미션을 수행해주세요."

    def _format_seconds(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return "알 수 없음"
        return f"{seconds:.2f}"