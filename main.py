import os
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
from ultralytics import YOLO

from hand_detector import HandDetector
from face_mesh_detector import FaceMeshDetector
from pose_detector import PoseDetector
from focus_analyzer import FocusBlurAnalyzer
from focus_action_manager import FocusActionManager, draw_focus_action_overlay
from mission_manager import MissionManager, draw_mission_overlay


# =========================
# 설정값
# =========================

MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0

CONF_THRESHOLD = 0.25
IMG_SIZE = 640

WINDOW_NAME = "Webcam Object Detection + Focus Mission"

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


def make_face_summary(face_infos):
    face_count = len(face_infos)

    if face_count == 0:
        return "none"

    return f"face: {face_count}"


def make_face_signature(face_infos):
    return len(face_infos)


def make_pose_summary(pose_infos):
    pose_count = len(pose_infos)

    if pose_count == 0:
        return "none"

    return f"pose: {pose_count}"


def make_pose_signature(pose_infos):
    return len(pose_infos)


def make_focus_summary(focus_analysis):
    score = focus_analysis["score"]
    level = focus_analysis["level"]
    face_direction = focus_analysis["face"]["direction"]
    eye_state = focus_analysis["face"]["eye_state"]

    active_reasons = focus_analysis["active_reasons"]

    if active_reasons:
        reasons_text = ", ".join(active_reasons)
    else:
        reasons_text = "none"

    return (
        f"score: {score}, level: {level}, "
        f"face: {face_direction}, eyes: {eye_state}, "
        f"reasons: {reasons_text}"
    )


def make_focus_action_summary(focus_action):
    state = focus_action["state"]
    level = focus_action["action_level"]
    duration = focus_action["duration"]

    if focus_action["active"]:
        return f"{state}, level: {level}, duration: {duration:.1f}s"

    if focus_action["recovered"]:
        return "recovered"

    return state


def make_mission_summary(mission_result):
    if mission_result["active"]:
        return (
            f"active, mission: {mission_result['mission_title']}, "
            f"progress: {int(mission_result['progress'] * 100)}%"
        )

    if mission_result["completed"]:
        return f"completed: {mission_result['mission_title']}"

    return "none"


def decide_event(
    stable_objects,
    hand_infos,
    face_infos,
    pose_infos,
    focus_analysis,
    focus_action,
    mission_result,
):
    counter = {}

    for obj in stable_objects:
        counter[obj["name"]] = obj["count"]

    person_count = counter.get("person", 0)
    laptop_count = counter.get("laptop", 0)
    cell_phone_count = counter.get("cell phone", 0)
    book_count = counter.get("book", 0)
    cup_count = counter.get("cup", 0)

    face_count = len(face_infos)
    pose_count = len(pose_infos)

    has_hand = len(hand_infos) > 0
    has_face = face_count > 0
    has_pose = pose_count > 0

    has_open_palm = any(hand.gesture == "open_palm" for hand in hand_infos)
    has_fist = any(hand.gesture == "fist" for hand in hand_infos)
    has_pointing = any(hand.gesture == "pointing" for hand in hand_infos)
    has_two_fingers = any(hand.gesture == "two_fingers" for hand in hand_infos)

    focus_score = focus_analysis["score"]
    focus_level = focus_analysis["level"]

    detected_text = ", ".join(
        f"{name}: {count}"
        for name, count in sorted(counter.items())
    )

    if not detected_text:
        detected_text = "none"

    if mission_result["active"]:
        main_event = f"Mission active: {mission_result['mission_title']}"
    elif mission_result["completed"]:
        main_event = f"Mission completed: {mission_result['mission_title']}"
    elif focus_action["active"]:
        main_event = f"Focus action active: {focus_action['message']}"
    elif focus_action["recovered"]:
        main_event = "Focus recovered"
    elif focus_score >= 75:
        main_event = f"High focus blur possibility ({focus_level})"
    elif focus_score >= 50:
        main_event = f"Focus warning ({focus_level})"
    elif has_open_palm:
        main_event = "Open palm gesture detected"
    elif has_fist:
        main_event = "Fist gesture detected"
    elif has_pointing:
        main_event = "Pointing gesture detected"
    elif has_two_fingers:
        main_event = "Two fingers gesture detected"
    elif face_count >= 2:
        main_event = "Multiple faces detected"
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
    elif has_hand and has_face and has_pose:
        main_event = "Face, hand, and pose detected"
    elif has_face and has_pose:
        main_event = "Face and pose detected"
    elif has_hand and has_pose:
        main_event = "Hand and pose detected"
    elif has_hand and has_face:
        main_event = "Face and hand detected"
    elif has_pose:
        main_event = "Pose detected"
    elif has_face:
        main_event = "Face detected"
    elif has_hand:
        main_event = "Hand detected"
    elif person_count == 1:
        main_event = "Person detected"
    elif stable_objects:
        main_event = "Objects detected"
    else:
        main_event = "Nothing stable detected"

    return f"EVENT: {main_event} / Stable Objects: {detected_text}"


def make_llm_prompt(
    stable_objects,
    event_text,
    hand_summary,
    face_summary,
    pose_summary,
    focus_summary,
    focus_action_summary,
    mission_summary,
):
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

현재 얼굴 랜드마크 추적 정보:
{face_summary}

현재 자세 랜드마크 추적 정보:
{pose_summary}

현재 집중 흐림 분석 정보:
{focus_summary}

현재 집중 저하 동작 상태:
{focus_action_summary}

현재 미션 상태:
{mission_summary}

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
    face_summary,
    pose_summary,
    focus_summary,
    focus_action_summary,
    mission_summary,
    event_text,
):
    lines = [
        f"FPS: {fps:.1f}",
        f"YOLO Raw: {raw_summary}",
        f"YOLO Stable: {stable_summary}",
        f"Hands: {hand_summary}",
        f"Face Mesh: {face_summary}",
        f"Pose: {pose_summary}",
        f"Focus: {focus_summary}",
        f"Action: {focus_action_summary}",
        f"Mission: {mission_summary}",
        event_text,
        "Q / ESC: Quit",
        "S: Save Screenshot",
    ]

    x = 10
    y = 30
    line_gap = 25

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
        )

    return frame


def save_screenshot(frame, prefix="capture"):
    SAVE_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"{prefix}_{timestamp}.jpg"

    success = cv2.imwrite(str(save_path), frame)

    if success:
        print(f"스크린샷 저장 완료: {save_path}")
    else:
        print("스크린샷 저장 실패")


def print_change_log(
    raw_summary,
    stable_summary,
    hand_summary,
    face_summary,
    pose_summary,
    focus_summary,
    focus_action_summary,
    mission_summary,
    focus_analysis,
    focus_action,
    mission_result,
    event_text,
    stable_objects,
):
    now_text = datetime.now().strftime("%H:%M:%S")

    print(f"[변화 감지] {now_text}")
    print(f"[YOLO Raw] {raw_summary}")
    print(f"[YOLO Stable] {stable_summary}")
    print(f"[Hands] {hand_summary}")
    print(f"[Face Mesh] {face_summary}")
    print(f"[Pose] {pose_summary}")
    print(f"[Focus] {focus_summary}")
    print(f"[Action] {focus_action_summary}")
    print(f"[Mission] {mission_summary}")

    print("[Focus Detail]")
    print(f"- face direction: {focus_analysis['face']['direction']}")
    print(f"- yaw ratio: {focus_analysis['face']['yaw_ratio']:.3f}")
    print(f"- head down ratio: {focus_analysis['face']['head_down_ratio']:.3f}")
    print(f"- eye state: {focus_analysis['face']['eye_state']}")
    print(f"- eye EAR: {focus_analysis['face']['eye_ear']:.3f}")
    print(f"- hand-face contact points: {focus_analysis['hand_face_contact']['contact_points']}")
    print(f"- pose movement: {focus_analysis['pose']['movement']:.3f}")
    print(f"- raw indicators: {focus_analysis['raw_indicators']}")
    print(f"- stable indicators: {focus_analysis['stable_indicators']}")

    print("[Action Detail]")
    print(f"- state: {focus_action['state']}")
    print(f"- active: {focus_action['active']}")
    print(f"- triggered: {focus_action['triggered']}")
    print(f"- recovered: {focus_action['recovered']}")
    print(f"- level: {focus_action['action_level']}")
    print(f"- message: {focus_action['message']}")
    print(f"- duration: {focus_action['duration']:.1f}s")
    print(f"- should_save_screenshot: {focus_action['should_save_screenshot']}")

    print("[Mission Detail]")
    print(f"- active: {mission_result['active']}")
    print(f"- completed: {mission_result['completed']}")
    print(f"- mission: {mission_result['mission_title']}")
    print(f"- progress: {mission_result['progress']:.2f}")
    print(f"- alarm_running: {mission_result['alarm_running']}")

    print(f"[이벤트] {event_text}")

    llm_prompt = make_llm_prompt(
        stable_objects=stable_objects,
        event_text=event_text,
        hand_summary=hand_summary,
        face_summary=face_summary,
        pose_summary=pose_summary,
        focus_summary=focus_summary,
        focus_action_summary=focus_action_summary,
        mission_summary=mission_summary,
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

    print("MediaPipe Face Mesh 로딩 중...")
    face_detector = FaceMeshDetector(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
        draw=True,
    )

    print("MediaPipe Pose 로딩 중...")
    pose_detector = PoseDetector(
        max_num_poses=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        draw=True,
    )

    print("집중 흐림 분석기 로딩 중...")
    focus_analyzer = FocusBlurAnalyzer(
        face_missing_hold_seconds=2.0,
        face_away_hold_seconds=1.5,
        head_down_hold_seconds=1.3,
        eyes_closed_hold_seconds=5.0,
        hand_face_hold_seconds=1.0,
        pose_unstable_hold_seconds=1.2,
    )

    print("집중 저하 동작 관리자 로딩 중...")
    focus_action_manager = FocusActionManager(
        enter_hold_seconds=0.8,
        recover_hold_seconds=2.0,
        action_cooldown_seconds=6.0,
        screenshot_after_seconds=8.0,
        warning_score_threshold=50,
        high_score_threshold=75,
        enable_auto_screenshot=True,
    )

    print("랜덤 미션 관리자 로딩 중...")
    mission_manager = MissionManager(
        mission_hold_seconds=1.0,
        mission_cooldown_seconds=2.0,
        enable_alarm=True,
    )

    cap = open_camera(CAMERA_INDEX)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        print("카메라 연결 상태를 확인하세요.")
        print("다른 프로그램이 카메라를 사용 중이면 종료한 뒤 다시 실행하세요.")
        hand_detector.close()
        face_detector.close()
        pose_detector.close()
        mission_manager.stop()
        return

    tracker = ObjectStateTracker(
        appear_threshold=APPEAR_THRESHOLD_SECONDS,
        disappear_threshold=DISAPPEAR_THRESHOLD_SECONDS,
    )

    print("웹캠 객체인식 + 집중 저하 감지 + 랜덤 미션 시작")
    print("종료: Q 또는 ESC")
    print("스크린샷 저장: S")
    print("집중 저하 시 랜덤 미션을 성공해야 경고음이 멈춥니다.")
    print()

    prev_time = time.time()
    last_logged_signature = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("웹캠 프레임을 읽지 못했습니다.")
                break

            frame = cv2.flip(frame, 1)

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

            hand_infos = hand_detector.detect(frame)
            hand_summary = hand_detector.make_summary(hand_infos)

            annotated_frame = draw_detections(frame, detections)
            annotated_frame = hand_detector.draw(annotated_frame, hand_infos)

            annotated_frame, face_infos = face_detector.detect(annotated_frame)
            face_summary = make_face_summary(face_infos)

            annotated_frame, pose_infos = pose_detector.detect(annotated_frame)
            pose_summary = make_pose_summary(pose_infos)

            focus_analysis = focus_analyzer.analyze(
                face_infos=face_infos,
                hand_infos=hand_infos,
                pose_infos=pose_infos,
                frame_shape=annotated_frame.shape,
            )
            focus_summary = make_focus_summary(focus_analysis)

            focus_action = focus_action_manager.update(focus_analysis)
            focus_action_summary = make_focus_action_summary(focus_action)

            mission_result = mission_manager.update(
                focus_action=focus_action,
                hand_infos=hand_infos,
                pose_infos=pose_infos,
                frame_shape=annotated_frame.shape,
            )
            mission_summary = make_mission_summary(mission_result)

            event_text = decide_event(
                stable_objects=stable_objects,
                hand_infos=hand_infos,
                face_infos=face_infos,
                pose_infos=pose_infos,
                focus_analysis=focus_analysis,
                focus_action=focus_action,
                mission_result=mission_result,
            )

            object_signature = make_stable_signature(stable_objects)
            hand_signature = hand_detector.make_signature(hand_infos)
            face_signature = make_face_signature(face_infos)
            pose_signature = make_pose_signature(pose_infos)
            focus_signature = focus_analyzer.make_signature(focus_analysis)
            action_signature = focus_action_manager.make_signature(focus_action)
            mission_signature = mission_manager.make_signature(mission_result)

            current_signature = {
                "objects": object_signature,
                "hands": hand_signature,
                "faces": face_signature,
                "poses": pose_signature,
                "focus": focus_signature,
                "action": action_signature,
                "mission": mission_signature,
            }

            if current_signature != last_logged_signature:
                print_change_log(
                    raw_summary=raw_summary,
                    stable_summary=stable_summary,
                    hand_summary=hand_summary,
                    face_summary=face_summary,
                    pose_summary=pose_summary,
                    focus_summary=focus_summary,
                    focus_action_summary=focus_action_summary,
                    mission_summary=mission_summary,
                    focus_analysis=focus_analysis,
                    focus_action=focus_action,
                    mission_result=mission_result,
                    event_text=event_text,
                    stable_objects=stable_objects,
                )

                last_logged_signature = current_signature

            if focus_action["should_save_screenshot"]:
                save_screenshot(annotated_frame, prefix="focus_blur")

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            annotated_frame = draw_status_panel(
                frame=annotated_frame,
                fps=fps,
                raw_summary=raw_summary,
                stable_summary=stable_summary,
                hand_summary=hand_summary,
                face_summary=face_summary,
                pose_summary=pose_summary,
                focus_summary=focus_summary,
                focus_action_summary=focus_action_summary,
                mission_summary=mission_summary,
                event_text=event_text,
            )

            annotated_frame = draw_focus_action_overlay(
                frame=annotated_frame,
                action_result=focus_action,
            )

            annotated_frame = draw_mission_overlay(
                frame=annotated_frame,
                mission_result=mission_result,
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
        mission_manager.stop()
        cap.release()
        hand_detector.close()
        face_detector.close()
        pose_detector.close()
        cv2.destroyAllWindows()
        print("프로그램 종료")


if __name__ == "__main__":
    main()