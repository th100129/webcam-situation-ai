from __future__ import annotations

import time

from focus.focus_action_manager import FocusActionManager


def main() -> None:
    manager = FocusActionManager()
    ok, message = manager.play_sound_test()
    print(message)
    if not ok:
        raise SystemExit(1)

    # 비동기 경고음 스레드가 예외를 기록할 시간을 조금 준다.
    time.sleep(0.8)
    status = manager.sound_status()
    print(f"[결과] {status}")
    if "오류" in status:
        raise SystemExit(1)
    print("[코드 점검 통과] Windows 경고음 호출을 정상적으로 요청했습니다.")
    print("단, 이 결과만으로 스피커·이어폰에서 실제로 들렸는지까지는 확인할 수 없습니다.")


if __name__ == "__main__":
    main()
