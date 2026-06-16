# prompt_templates.py


MISSION_PROMPT_TEMPLATE = """
너는 웹캠 기반 집중 관리 시스템의 피드백 문구를 작성하는 도우미야.

상황:
- 집중 상태: {focus_state}
- 감지 사유: {reason_label}
- 감지 상세: {reason_detail}
- 선택된 미션: {mission_label}
- 미션 설명: {mission_description}

요청:
사용자가 부담 없이 따라 할 수 있도록 짧은 미션 안내 문구를 작성해줘.

조건:
- 한국어로 작성
- 1문장
- 비난하거나 혼내는 말투 금지
- 너무 길지 않게
- 미션 동작이 명확히 드러나야 함
"""


SUCCESS_PROMPT_TEMPLATE = """
너는 웹캠 기반 집중 관리 시스템의 피드백 문구를 작성하는 도우미야.

상황:
- 사용자가 미션을 성공함
- 성공한 미션: {mission_label}
- 반응 시간: {reaction_time}초

요청:
사용자에게 보여줄 짧은 성공 피드백 문구를 작성해줘.

조건:
- 한국어로 작성
- 1문장
- 칭찬 + 다시 집중 유도
- 과하게 오글거리지 않게
"""


FAIL_PROMPT_TEMPLATE = """
너는 웹캠 기반 집중 관리 시스템의 피드백 문구를 작성하는 도우미야.

상황:
- 사용자가 미션을 제한 시간 안에 성공하지 못함
- 실패한 미션: {mission_label}
- 제한 시간: {timeout_seconds}초

요청:
사용자에게 보여줄 짧은 실패 안내 문구를 작성해줘.

조건:
- 한국어로 작성
- 1문장
- 비난 금지
- 다시 시도하거나 화면을 확인하도록 안내
"""


SESSION_SUMMARY_PROMPT_TEMPLATE = """
너는 웹캠 기반 집중 관리 시스템의 세션 요약 리포트를 작성하는 도우미야.

세션 데이터:
- 전체 이벤트 수: {total_events}
- 집중 흐림 감지 횟수: {focus_alert_count}
- 미션 시도 횟수: {mission_attempt_count}
- 미션 성공 횟수: {mission_success_count}
- 미션 실패 횟수: {mission_fail_count}
- 미션 성공률: {mission_success_rate}%
- 평균 반응 시간: {avg_reaction_time}초
- 평균 FPS: {avg_fps}
- 오탐 횟수: {false_positive_count}
- 미탐 횟수: {false_negative_count}

요청:
위 데이터를 바탕으로 사용자가 이해하기 쉬운 세션 요약 문장을 작성해줘.

조건:
- 한국어로 작성
- 3~5문장
- 너무 딱딱하지 않게
- 성과와 개선점을 함께 언급
"""