from __future__ import annotations

import sys
import time

import cv2

from config import (
    ANALYSIS_MAX_HEIGHT,
    ANALYSIS_MAX_WIDTH,
    APPEAR_THRESHOLD_SECONDS,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAPTURE_DIR,
    DEBUG_LANDMARKS_DEFAULT,
    DISAPPEAR_THRESHOLD_SECONDS,
    DISPLAY_MAX_HEIGHT,
    DISPLAY_MAX_WIDTH,
    ENABLE_OBJECT_DETECTION,
    FACE_MODEL_PATH,
    HAND_MODEL_PATH,
    LOG_DIR,
    MIRROR_VIEW,
    OBJECT_DETECTION_INTERVAL,
    POSE_MODEL_PATH,
    TARGET_CLASSES,
    WINDOW_NAME,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL_PATH,
)
from detectors.face_mesh_detector import FaceMeshDetector
from detectors.hand_detector import HandDetector
from detectors.object_detector import ObjectDetector
from detectors.pose_detector import PoseDetector
from feedback.llm_manager import LLMManager
from focus.focus_action_manager import FocusActionManager
from focus.focus_analyzer import FocusAnalyzer
from mission.mission_manager import MissionManager
from tracking.object_state_tracker import ObjectStateTracker
from ui.particle_system import ParticleSystem
from ui.renderer import (
    draw_completion_banner,
    draw_dashboard,
    draw_focus_status,
    draw_mission_overlay,
    draw_toast,
    draw_vision_overlays,
)
from utils.camera import camera_resolution, open_camera
from utils.logging_utils import print_change_log
from utils.metrics_tracker import MetricsTracker
from utils.report_generator import ReportGenerator
from utils.screenshot import save_screenshot
from utils.session_logger import SessionLogger


def resize_to_max(frame, max_width: int, max_height: int, allow_upscale: bool) -> object:
    """비율을 유지하며 최대 크기에 맞춘다. 확대 시 INTER_CUBIC으로 덜 뭉개지게 처리한다."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    if abs(scale - 1.0) < 0.01:
        return frame.copy()

    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)


def particle_origin(faces, width: int, height: int) -> tuple[int, int]:
    """성공한 미션의 파티클이 얼굴 근처에서 터지도록 시작 좌표를 계산한다."""
    if faces:
        landmarks = faces[0].landmarks
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        center_x = int(((min(xs) + max(xs)) / 2.0) * width)
        # 얼굴 한가운데보다 살짝 위에서 터지게 해 표정을 오래 가리지 않는다.
        center_y = int(((min(ys) + max(ys)) / 2.0) * height - height * 0.06)
        return center_x, max(80, center_y)
    return width // 2, max(100, int(height * 0.40))


def build_event_text(focus_analysis: dict, mission_result: dict, faces, hands, poses, stable_objects) -> str:
    if mission_result["active"]:
        return f"미션 진행 중: {mission_result['mission_title']}"
    if mission_result["completed"]:
        return f"미션 성공: {mission_result['mission_title']}"
    if focus_analysis["active_reasons"]:
        return "집중 흐림 감지: " + ", ".join(focus_analysis["active_reason_labels"])
    if faces and hands and poses:
        return "얼굴·손·자세를 추적 중"
    if faces:
        return "얼굴 추적 중"
    if stable_objects:
        return "안정 객체 추적 중"
    return "대기 중"


def create_detectors():
    """감지는 분석 프레임에서만 하고, 오버레이는 확대된 출력 프레임에서 다시 그린다."""
    face_detector = FaceMeshDetector(FACE_MODEL_PATH, draw=False)
    hand_detector = HandDetector(HAND_MODEL_PATH, draw=False)
    pose_detector = PoseDetector(POSE_MODEL_PATH, draw=False)

    object_detector = None
    if ENABLE_OBJECT_DETECTION:
        try:
            object_detector = ObjectDetector(
                model_path=YOLO_MODEL_PATH,
                confidence=YOLO_CONFIDENCE,
                image_size=YOLO_IMAGE_SIZE,
                target_classes=TARGET_CLASSES,
            )
        except Exception as error:
            print(f"[경고] YOLO 객체 인식은 비활성화됩니다: {error}")
    return face_detector, hand_detector, pose_detector, object_detector


def start_mission(mission_manager, action_manager, llm_manager, metrics, logger, reason: str | None, manual: bool = False):
    mission_result = mission_manager.start(reason)
    metrics.mission_started()
    logger.log(
        "mission_started",
        {"reason": reason, "manual": manual, "mission": mission_result},
    )
    action_manager.maybe_play_warning(True)
    message = llm_manager.mission_message(mission_result)
    return mission_result, message


def main() -> None:
    try:
        face_detector, hand_detector, pose_detector, object_detector = create_detectors()
    except Exception as error:
        print("\n[시작 실패]")
        print(error)
        print("\n아래 명령을 먼저 실행한 뒤 다시 시작하세요.")
        print("  python -m pip install -r requirements.txt")
        print("  python download_models.py")
        sys.exit(1)

    camera = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    if not camera.isOpened():
        print(f"카메라 {CAMERA_INDEX}번을 열지 못했습니다.")
        sys.exit(1)

    actual_width, actual_height, actual_fps = camera_resolution(camera)
    print("\nFocus Trigger 시작")
    print(f"- 카메라 요청: {CAMERA_WIDTH}x{CAMERA_HEIGHT} / 실제: {actual_width}x{actual_height} ({actual_fps:.0f} FPS)")
    print("- 눈을 5초 감거나, 고개·얼굴을 3초 이탈하면 경고음과 미션이 시작됩니다.")
    print("- M: 랜덤 미션 수동 테스트 / B: 경고음 수동 테스트 / D: 랜드마크 디버그")
    print("- Q 또는 ESC 종료 / S 스크린샷\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    tracker = ObjectStateTracker(APPEAR_THRESHOLD_SECONDS, DISAPPEAR_THRESHOLD_SECONDS)
    focus_analyzer = FocusAnalyzer()
    focus_action_manager = FocusActionManager()
    mission_manager = MissionManager()
    llm_manager = LLMManager()
    particles = ParticleSystem()
    logger = SessionLogger(LOG_DIR)
    metrics = MetricsTracker()

    latest_detections = []
    frame_index = 0
    last_frame_time = time.monotonic()
    last_logged_signature = None
    debug_enabled = DEBUG_LANDMARKS_DEFAULT
    mission_feedback = ""
    completion_message = ""
    completion_banner_until = 0.0
    toast_message = ""
    toast_banner_until = 0.0
    last_mission_completed_signature = None
    pending_success_particles = False

    try:
        while True:
            ok, camera_frame = camera.read()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
                break
            if MIRROR_VIEW:
                camera_frame = cv2.flip(camera_frame, 1)

            now = time.monotonic()
            delta = max(now - last_frame_time, 1e-6)
            last_frame_time = now
            fps = 1.0 / delta
            frame_index += 1
            metrics.update_frame(fps)

            # 실제 카메라 해상도와 출력 해상도를 분리한다.
            # 감지는 너무 큰 프레임에서 하지 않아 FPS를 방어하고, 출력은 가능한 큰 원본에 유지한다.
            analysis_frame = resize_to_max(camera_frame, ANALYSIS_MAX_WIDTH, ANALYSIS_MAX_HEIGHT, allow_upscale=False)
            analysis_height, analysis_width = analysis_frame.shape[:2]

            _, faces = face_detector.detect(analysis_frame)
            _, hands = hand_detector.detect(analysis_frame)
            _, poses = pose_detector.detect(analysis_frame)

            if object_detector is not None and frame_index % OBJECT_DETECTION_INTERVAL == 0:
                latest_detections = object_detector.detect(analysis_frame)
                tracker.update(latest_detections)

            stable_objects = tracker.get_stable_objects()
            focus_analysis = focus_analyzer.analyze(faces, hands, poses)
            action_result = focus_action_manager.update(focus_analysis, mission_manager.active)

            if action_result["should_start_mission"] and not mission_manager.active:
                _, mission_feedback = start_mission(
                    mission_manager,
                    focus_action_manager,
                    llm_manager,
                    metrics,
                    logger,
                    action_result["primary_reason"],
                )

            mission_result = mission_manager.update(focus_analysis, hands, poses)
            focus_action_manager.maybe_play_warning(mission_result["active"])

            if mission_result["timed_out"]:
                mission_feedback = llm_manager.retry_message(mission_result)
                metrics.mission_timed_out()
                logger.log("mission_timeout_retry", mission_result)

            completed_signature = (
                mission_result["completed"],
                mission_result["mission_id"],
                mission_result["reaction_time"],
            )
            if mission_result["completed"] and completed_signature != last_mission_completed_signature:
                last_mission_completed_signature = completed_signature
                reaction_time = float(mission_result["reaction_time"] or 0.0)
                focus_action_manager.mission_succeeded()
                metrics.mission_completed(reaction_time)
                completion_message = llm_manager.success_message(mission_result)
                completion_banner_until = time.monotonic() + 2.8
                pending_success_particles = True
                mission_feedback = ""
                logger.log("mission_completed", mission_result)
            elif not mission_result["completed"]:
                last_mission_completed_signature = None

            # 출력을 먼저 확대한 뒤 박스·문자·랜드마크를 그려서 흐릿해지는 문제를 없앤다.
            display_frame = resize_to_max(camera_frame, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT, allow_upscale=True)
            display_height, display_width = display_frame.shape[:2]
            if pending_success_particles:
                particles.celebrate(
                    particle_origin(faces, display_width, display_height),
                    (display_width, display_height),
                )
                pending_success_particles = False

            # 파티클은 얼굴·객체 박스보다 먼저 그려서 인식 결과를 가리지 않게 한다.
            particles.update_and_draw(display_frame, delta)
            draw_vision_overlays(
                display_frame,
                faces,
                hands,
                poses,
                latest_detections,
                (analysis_width, analysis_height),
                debug_enabled,
            )

            raw_summary = ObjectDetector.summary(latest_detections) if object_detector else "disabled"
            stable_summary = tracker.summary(stable_objects)
            event_text = build_event_text(focus_analysis, mission_result, faces, hands, poses, stable_objects)

            display_frame = draw_dashboard(
                display_frame,
                fps,
                focus_analysis,
                mission_result,
                stable_summary,
                raw_summary,
                focus_action_manager.sound_status(),
            )
            display_frame = draw_focus_status(display_frame, focus_analysis, action_result)
            display_frame = draw_mission_overlay(display_frame, mission_result, mission_feedback)
            display_frame = draw_completion_banner(
                display_frame,
                completion_message,
                completion_banner_until - time.monotonic(),
            )
            display_frame = draw_toast(
                display_frame,
                toast_message,
                toast_banner_until - time.monotonic(),
                color=(150, 80, 0),
            )

            signature = (
                tuple((item["name"], item["count"]) for item in stable_objects),
                FocusAnalyzer.signature(focus_analysis),
                MissionManager.signature(mission_result),
                action_result["state"],
            )
            if signature != last_logged_signature:
                print_change_log(focus_analysis, mission_result, event_text)
                logger.log(
                    "state_changed",
                    {
                        "event": event_text,
                        "focus": focus_analysis,
                        "mission": mission_result,
                        "stable_objects": stable_objects,
                    },
                )
                last_logged_signature = signature

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_screenshot(display_frame, CAPTURE_DIR)
                print(f"[스크린샷 저장] {path}")
                logger.log("screenshot_saved", {"path": str(path)})
            if key == ord("d"):
                debug_enabled = not debug_enabled
                toast_message = f"랜드마크 디버그 {'켜짐' if debug_enabled else '꺼짐'}"
                toast_banner_until = time.monotonic() + 2.0
                print(f"[랜드마크 디버그] {'켜짐' if debug_enabled else '꺼짐'}")
            if key == ord("b"):
                ok_sound, message = focus_action_manager.play_sound_test()
                toast_message = "사운드 테스트 요청 전송" if ok_sound else "사운드 테스트 실패"
                toast_banner_until = time.monotonic() + 3.0
                print(f"[사운드 테스트] {message}")
            if key == ord("m"):
                if mission_manager.active:
                    toast_message = "이미 미션이 진행 중입니다. 성공하면 경고음이 멈춥니다."
                    toast_banner_until = time.monotonic() + 2.5
                else:
                    _, mission_feedback = start_mission(
                        mission_manager,
                        focus_action_manager,
                        llm_manager,
                        metrics,
                        logger,
                        "manual_test",
                        manual=True,
                    )
                    toast_message = "수동 미션 테스트를 시작했습니다."
                    toast_banner_until = time.monotonic() + 2.2
                    print("[수동 테스트] 랜덤 미션 시작")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        for detector in (face_detector, hand_detector, pose_detector):
            try:
                detector.close()
            except Exception:
                pass
        summary = metrics.summary()
        report_path = ReportGenerator.save(LOG_DIR, summary)
        logger.log("session_end", {"summary": summary, "report_path": str(report_path)})
        print("\n세션 종료")
        print(f"로그: {logger.path}")
        print(f"리포트: {report_path}")
        print(summary)


if __name__ == "__main__":
    main()
