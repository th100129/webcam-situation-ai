import os
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
from ultralytics import YOLO

from hand_detector import HandDetector


# =========================
# 설정값
# =========================

MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0

CONF_THRESHOLD = 0.25
IMG_SIZE = 640

WINDOW_NAME = "Webcam Object Detection + MediaPipe Hands"

TARGET_CLASSES = None

APPEAR_THRESHOLD_SECONDS = 0.8
DISAPPEAR_THRESHOLD_SECONDS = 2.0

SAVE_DIR = Path("captures")


# =========================
# 객체 상태 추적 클래스
# =========================

class ObjectStateTracker:
    def __init__(self, appear_threshold=0.8, disappear_threshold=2.0):
        self.appear_threshold = appear_threshold
        self.disappear_threshold = disappear_threshold
        self.states = {}

    def update(self, detections):
        now = time.time()

        current_counter = Counter(det["name"] for det in detections)

        confidence_by_name = {}
        for det in detections:
            name = det["name"]
            confidence = det["confidence"]

            if name not in confidence_by_name:
                confidence_by_name[name] = confidence
            else:
                confidence_by_name[name] = max(confidence_by_name[name], confidence)

        current_names = set(current_counter.keys())

        for name in current_names:
            count = current_counter[name]
            max_confidence = confidence_by_name.get(name, 0.0)

            if name not in self.states:
                self.states[name] = {
                    "first_seen": now,
                    "last_seen": now,
                    "stable": False,
                    "count": count,
                    "max_confidence": max_confidence,
                }
            else:
                self.states[name]["last_seen"] = now
                self.states[name]["count"] = count
                self.states[name]["max_confidence"] = max_confidence

            visible_duration = now - self.states[name]["first_seen"]

            if visible_duration >= self.appear_threshold:
                self.states[name]["stable"] = True

        remove_targets = []

        for name, state in self.states.items():
            missing_duration = now - state["last_seen"]

            if missing_duration >= self.disappear_threshold:
                remove_targets.append(name)

        for name in remove_targets:
            del self.states[name]

    def get_stable_objects(self):
        now = time.time()
        stable_objects = []

        for name, state in self.states.items():
            if not state["stable"]:
                continue

            missing_duration = now - state["last_seen"]

            if missing_duration < self.disappear_threshold:
                stable_objects.append({
                    "name": name,
                    "count": state["count"],
                    "duration": now - state["first_seen"],
                    "max_confidence": state["max_confidence"],
                    "missing_duration": missing_duration,
                })

        stable_objects.sort(key=lambda item: item["name"])

        return stable_objects


# =========================
# 유틸 함수
# =========================

def open_camera(camera_index):
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))

    return names[class_id]


def extract_detections(result, target_classes=None):
    detections = []

    if result.boxes is None:
        return detections

    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = get_class_name(names, class_id)
        confidence = float(box.conf[0].item())

        if target_classes is not None and class_name not in target_classes:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

        detections.append({
            "class_id": class_id,
            "name": class_name,
            "confidence": confidence,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        })

    return detections


def draw_detections(frame, detections):
    annotated = frame.copy()

    for det in detections:
        name = det["name"]
        confidence = det["confidence"]
        bbox = det["bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        label = f"{name} {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated


def make_raw_summary(detections):
    if not detections:
        return "none"

    counter = Counter(det["name"] for det in detections)

    parts = []
    for name, count in counter.most_common():
        parts.append(f"{name}: {count}")

    return ", ".join(parts)


def make_stable_summary(stable_objects):
    if not stable_objects:
        return "none"

    parts = []

    for obj in stable_objects:
        name = obj["name"]
        count = obj["count"]
        duration = obj["duration"]
        parts.append(f"{name}: {count} ({duration:.1f}s)")

    return ", ".join(parts)


def make_stable_signature(stable_objects):
    signature_items = []

    for obj in stable_objects:
        signature_items.append((obj["name"], obj["count"]))

    signature_items.sort()

    return tuple(signature_items)


def decide_event(stable_objects, hand_infos):
    counter = {}

    for obj in stable_objects:
        counter[obj["name"]] = obj["count"]

    person_count = counter.get("person", 0)
    laptop_count = counter.get("laptop", 0)
    cell_phone_count = counter.get("cell phone", 0)
    book_count = counter.get("book", 0)
    cup_count = counter.get("cup", 0)

    has_hand = len(hand_infos) > 0
    has_open_palm = any(hand.gesture == "open_palm" for hand in hand_infos)
    has_fist = any(hand.gesture == "fist" for hand in hand_infos)
    has_pointing = any(hand.gesture == "pointing" for hand in hand_infos)
    has_two_fingers = any(hand.gesture == "two_fingers" for hand in hand_infos)

    detected_text = ", ".join(
        f"{name}: {count}"
        for name, count in sorted(counter.items())
    )

    if not detected_text:
        detected_text = "none"

    if has_open_palm:
        main_event = "Open palm gesture detected"
    elif has_fist:
        main_event = "Fist gesture detected"
    elif has_pointing:
        main_event = "Pointing gesture detected"
    elif has_two_fingers:
        main_event = "Two fingers gesture detected"
    elif person_count >= 2:
        main_event = "Multiple people detected"
    elif person_count == 1 and laptop_count >= 1:
        main_event = "Working or studying situation"
    elif person_count == 1 and cell_phone_count >= 1:
        main_event = "Person using phone"
    elif person_count == 1 and book_count >= 1:
        main_event = "Reading situation"
    elif person_count == 1 and cup_count >= 1:
        main_event = "Person with drink"
    elif has_hand:
        main_event = "Hand detected"
    elif person_count == 1:
        main_event = "Person detected"
    elif stable_objects:
        main_event = "Objects detected"
    else:
        main_event = "Nothing stable detected"

    return f"EVENT: {main_event} / Stable Objects: {detected_text}"


def make_llm_prompt(stable_objects, event_text, hand_summary):
    if not stable_objects:
        object_text = "안정적으로 감지된 객체가 없습니다."
    else:
        object_lines = []

        for obj in stable_objects:
            object_lines.append(
                f"- {obj['name']}: {obj['count']}개, 약 {obj['duration']:.1f}초 동안 감지됨"
            )

        object_text = "\n".join(object_lines)

    prompt = f"""
현재 웹캠에서 감지된 객체 정보는 다음과 같습니다.

{object_text}

현재 손 추적 정보:
{hand_summary}

현재 이벤트 판단:
{event_text}

위 정보를 바탕으로 사용자가 이해하기 쉽게 현재 상황을 한두 문장으로 설명하세요.
"""

    return prompt.strip()


def draw_status_panel(
    frame,
    fps,
    raw_summary,
    stable_summary,
    hand_summary,
    event_text,
):
    lines = [
        f"FPS: {fps:.1f}",
        f"YOLO Raw: {raw_summary}",
        f"YOLO Stable: {stable_summary}",
        f"Hands: {hand_summary}",
        event_text,
        "Log: only when object/hand state changes",
        "Q / ESC: Quit",
        "S: Save Screenshot",
    ]

    x = 10
    y = 30
    line_gap = 28

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

    return frame


def save_screenshot(frame):
    SAVE_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"capture_{timestamp}.jpg"

    success = cv2.imwrite(str(save_path), frame)

    if success:
        print(f"스크린샷 저장 완료: {save_path}")
    else:
        print("스크린샷 저장 실패")


def print_change_log(
    raw_summary,
    stable_summary,
    hand_summary,
    event_text,
    stable_objects,
):
    now_text = datetime.now().strftime("%H:%M:%S")

    print(f"[변화 감지] {now_text}")
    print(f"[YOLO Raw] {raw_summary}")
    print(f"[YOLO Stable] {stable_summary}")
    print(f"[Hands] {hand_summary}")
    print(f"[이벤트] {event_text}")

    llm_prompt = make_llm_prompt(
        stable_objects=stable_objects,
        event_text=event_text,
        hand_summary=hand_summary,
    )

    print("[LLM 프롬프트 예시]")
    print(llm_prompt)
    print("-" * 60)


# =========================
# 메인 실행
# =========================

def main():
    print("YOLO 모델 로딩 중...")
    model = YOLO(MODEL_PATH)

    print("MediaPipe Hands 로딩 중...")
    hand_detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = open_camera(CAMERA_INDEX)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        print("카메라 연결 상태를 확인하세요.")
        print("다른 프로그램이 카메라를 사용 중이면 종료한 뒤 다시 실행하세요.")
        hand_detector.close()
        return

    tracker = ObjectStateTracker(
        appear_threshold=APPEAR_THRESHOLD_SECONDS,
        disappear_threshold=DISAPPEAR_THRESHOLD_SECONDS,
    )

    print("웹캠 객체인식 + 손 추적 시작")
    print("종료: Q 또는 ESC")
    print("스크린샷 저장: S")
    print("로그 출력: Stable 객체 또는 손 상태가 바뀔 때만 출력")
    print()

    prev_time = time.time()
    last_logged_signature = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("웹캠 프레임을 읽지 못했습니다.")
                break

            # YOLO 객체 인식
            result = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False,
            )[0]

            detections = extract_detections(
                result=result,
                target_classes=TARGET_CLASSES,
            )

            tracker.update(detections)
            stable_objects = tracker.get_stable_objects()

            raw_summary = make_raw_summary(detections)
            stable_summary = make_stable_summary(stable_objects)

            # MediaPipe Hands
            hand_infos = hand_detector.detect(frame)
            hand_summary = hand_detector.make_summary(hand_infos)

            event_text = decide_event(
                stable_objects=stable_objects,
                hand_infos=hand_infos,
            )

            object_signature = make_stable_signature(stable_objects)
            hand_signature = hand_detector.make_signature(hand_infos)

            current_signature = {
                "objects": object_signature,
                "hands": hand_signature,
            }

            if current_signature != last_logged_signature:
                print_change_log(
                    raw_summary=raw_summary,
                    stable_summary=stable_summary,
                    hand_summary=hand_summary,
                    event_text=event_text,
                    stable_objects=stable_objects,
                )

                last_logged_signature = current_signature

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # 화면 표시
            annotated_frame = draw_detections(frame, detections)
            annotated_frame = hand_detector.draw(annotated_frame, hand_infos)

            annotated_frame = draw_status_panel(
                frame=annotated_frame,
                fps=fps,
                raw_summary=raw_summary,
                stable_summary=stable_summary,
                hand_summary=hand_summary,
                event_text=event_text,
            )

            cv2.imshow(WINDOW_NAME, annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("s"):
                save_screenshot(annotated_frame)

    except KeyboardInterrupt:
        print("\nCtrl+C 입력으로 종료합니다.")

    finally:
        cap.release()
        hand_detector.close()
        cv2.destroyAllWindows()
        print("프로그램 종료")


if __name__ == "__main__":
    main()