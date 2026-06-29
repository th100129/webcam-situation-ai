from __future__ import annotations


MISSION_PROMPT = """사용자의 집중 흐림 사유는 '{reason}'이고, 지금 해야 할 미션은 '{mission_title}'입니다.
미션 설명: {instruction}
사용자에게 부담스럽지 않고 짧은 한국어 문장으로 미션을 안내하세요."""

SUCCESS_PROMPT = """사용자가 '{mission_title}' 미션을 {reaction_time:.1f}초 만에 성공했습니다.
따뜻하고 짧은 한국어 피드백을 만들어주세요."""

RETRY_PROMPT = """사용자가 '{mission_title}' 미션의 제한 시간 안에 성공하지 못했습니다.
경고하지 말고, 같은 미션을 다시 해보도록 짧고 친절한 한국어로 안내하세요."""
