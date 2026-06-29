from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import KOREAN_FONT_CANDIDATES
from focus.focus_analyzer import FocusAnalyzer


class KoreanTextRenderer:
    """OpenCV의 한글 깨짐을 피하는 Pillow 기반 렌더러."""

    def __init__(self) -> None:
        self.font_path = self._find_font()
        self.cache: Dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    @staticmethod
    def _find_font() -> Path | None:
        for path in KOREAN_FONT_CANDIDATES:
            if path.exists():
                return path
        return None

    def font(self, size: int):
        if size not in self.cache:
            self.cache[size] = ImageFont.truetype(str(self.font_path), size) if self.font_path else ImageFont.load_default()
        return self.cache[size]

    def draw(self, frame, text: str, position: Tuple[int, int], size: int, color=(255, 255, 255)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        ImageDraw.Draw(image).text(position, text, font=self.font(size), fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def draw_lines(self, frame, lines: List[str], position: Tuple[int, int], size: int, line_gap: int, color=(255, 255, 255)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        x, y = position
        font = self.font(size)
        for index, line in enumerate(lines):
            draw.text((x, y + index * line_gap), line, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


TEXT = KoreanTextRenderer()


FACE_CONTOUR = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 152, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def _transparent_box(frame, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], alpha: float = 0.72) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _norm_point(point, width: int, height: int) -> tuple[int, int]:
    return int(point[0] * width), int(point[1] * height)


def _norm_bbox(points, width: int, height: int) -> tuple[int, int, int, int]:
    xs = [int(point[0] * width) for point in points]
    ys = [int(point[1] * height) for point in points]
    return max(0, min(xs)), max(0, min(ys)), min(width - 1, max(xs)), min(height - 1, max(ys))


def _label_box(frame, text: str, x: int, y: int, color: Tuple[int, int, int], scale: float = 0.55) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
    y = max(text_height + 8, y)
    cv2.rectangle(frame, (x, y - text_height - 8), (x + text_width + 10, y + baseline - 3), color, -1)
    cv2.putText(frame, text, (x + 5, y - 4), font, scale, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_polyline(frame, landmarks, indices: List[int], width: int, height: int, color, thickness: int = 1) -> None:
    points = np.array([_norm_point(landmarks[index], width, height) for index in indices], dtype=np.int32)
    cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)


def draw_vision_overlays(
    frame,
    faces,
    hands,
    poses,
    detections,
    analysis_size: tuple[int, int],
    debug_enabled: bool,
) -> object:
    """검출 결과를 확대된 출력 프레임에 다시 그린다.

    감지는 작은 분석 프레임에서 수행하고, 박스·문자·랜드마크는 확대 후 그려서
    선과 텍스트가 흐려지는 문제를 막는다.
    """
    height, width = frame.shape[:2]
    source_width, source_height = analysis_size
    line = 2 if width < 1100 else 3

    for face in faces:
        x1, y1, x2, y2 = _norm_bbox(face.landmarks, width, height)
        color = (0, 70, 255) if face.eyes_closed else (35, 230, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line, cv2.LINE_AA)
        eye_text = "closed" if face.eyes_closed else "open"
        label = f"FACE  {face.direction}  |  eyes {eye_text}  |  EAR {face.ear:.2f}"
        _label_box(frame, label, x1, y1 - 5, color, scale=0.52 if width < 1100 else 0.62)

        if debug_enabled:
            _draw_polyline(frame, face.landmarks, FACE_CONTOUR, width, height, (0, 220, 255), 1)
            _draw_polyline(frame, face.landmarks, LEFT_EYE, width, height, (0, 220, 255), 1)
            _draw_polyline(frame, face.landmarks, RIGHT_EYE, width, height, (0, 220, 255), 1)
            _draw_polyline(frame, face.landmarks, MOUTH, width, height, (0, 220, 255), 1)
            for index in [1, 10, 152, 61, 291]:
                x, y = _norm_point(face.landmarks[index], width, height)
                cv2.circle(frame, (x, y), 2, (0, 220, 255), -1, cv2.LINE_AA)

    for hand in hands:
        x1, y1, x2, y2 = _norm_bbox(hand.landmarks, width, height)
        color = (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line, cv2.LINE_AA)
        _label_box(frame, f"HAND  {hand.gesture}  |  {hand.finger_count} fingers", x1, y1 - 5, color, scale=0.50 if width < 1100 else 0.60)
        if debug_enabled:
            pixels = [_norm_point(point, width, height) for point in hand.landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, pixels[start], pixels[end], (255, 180, 0), 2, cv2.LINE_AA)
            for point in pixels:
                cv2.circle(frame, point, 2, (0, 230, 255), -1, cv2.LINE_AA)

    for pose in poses:
        if debug_enabled:
            pixels = [_norm_point(point, width, height) for point in pose.landmarks]
            for start, end in POSE_CONNECTIONS:
                cv2.line(frame, pixels[start], pixels[end], (255, 0, 255), 2, cv2.LINE_AA)
            label = f"POSE  tilt {pose.shoulder_tilt:.2f}  motion {pose.motion:.2f}"
            x1, y1, _, _ = _norm_bbox(pose.landmarks, width, height)
            _label_box(frame, label, x1, y1 - 5, (255, 0, 255), scale=0.48 if width < 1100 else 0.58)

    scale_x = width / max(source_width, 1)
    scale_y = height / max(source_height, 1)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        color = (75, 235, 75)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line, cv2.LINE_AA)
        _label_box(frame, f"{detection['name']} {detection['confidence']:.2f}", x1, y1 - 5, color, scale=0.50 if width < 1100 else 0.60)

    return frame


def draw_dashboard(frame, fps: float, focus_analysis: Dict, mission_result: Dict, stable_summary: str, object_summary: str, sound_status: str) -> object:
    height, width = frame.shape[:2]
    box_width = min(width - 28, 520)
    box_height = 154
    _transparent_box(frame, 14, 14, 14 + box_width, 14 + box_height, (20, 20, 20), 0.66)

    face = focus_analysis["face"]
    lines = [
        f"FPS  {fps:.1f}",
        f"Focus  {focus_analysis['score']}  ({focus_analysis['level']})",
        f"Face  {face['direction']}  |  eyes {face['eye_state']}  |  EAR {face['ear']:.2f}",
        f"Objects  {object_summary}  |  stable {stable_summary}",
        "M: mission test  |  B: sound test  |  D: landmark debug",
        "Q / ESC: exit  |  S: screenshot",
    ]
    for index, line in enumerate(lines):
        scale = 0.63 if index == 0 else 0.48
        thickness = 2 if index == 0 else 1
        cv2.putText(frame, line, (28, 42 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), thickness, cv2.LINE_AA)

    sound_color = (80, 225, 80) if "오류" not in sound_status else (0, 80, 255)
    cv2.putText(frame, sound_status, (28, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sound_color, 1, cv2.LINE_AA)
    return frame


def _tracking_text(focus_analysis: Dict) -> str:
    raw = focus_analysis["raw_indicators"]
    durations = focus_analysis["durations"]
    candidates = []
    for reason, active in raw.items():
        if active:
            label = FocusAnalyzer.REASON_LABELS[reason]
            candidates.append(f"{label} {durations[reason]:.1f}/{FocusAnalyzer.SIGNAL_SECONDS[reason]:.0f}s")
    return "판정 중: " + " | ".join(candidates) if candidates else ""


def draw_focus_status(frame, focus_analysis: Dict, action_result: Dict, last_message: str = "") -> object:
    tracking_text = _tracking_text(focus_analysis)
    if action_result["state"] == "cooldown":
        text = f"미션 성공! 잠시 쉬어갈게요. ({action_result['cooldown_left']:.1f}s)"
        color = (70, 220, 70)
    elif focus_analysis["active_reasons"]:
        labels = ", ".join(focus_analysis["active_reason_labels"])
        text = f"집중 흐림 감지: {labels}"
        color = (0, 80, 255)
    elif tracking_text:
        text = tracking_text
        color = (0, 200, 255)
    elif last_message:
        text = last_message
        color = (0, 215, 255)
    else:
        return frame

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = 14, height - 62, min(width - 14, 950), height - 14
    _transparent_box(frame, x1, y1, x2, y2, (20, 20, 20), 0.66)
    return TEXT.draw(frame, text, (x1 + 16, y1 + 12), size=22, color=color)


def draw_mission_overlay(frame, mission_result: Dict, message: str = "") -> object:
    if not mission_result["active"]:
        return frame

    height, width = frame.shape[:2]
    box_width = min(width - 50, 730)
    box_height = 200
    x1 = (width - box_width) // 2
    y1 = height - box_height - 84
    x2, y2 = x1 + box_width, y1 + box_height
    _transparent_box(frame, x1, y1, x2, y2, (40, 20, 90), 0.80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 100, 255), 3, cv2.LINE_AA)

    reason = FocusAnalyzer.REASON_LABELS.get(mission_result["reason"], "테스트 미션")
    progress = int(mission_result["progress"] * 100)
    lines = [
        "집중 리셋 미션 — 경고음은 성공할 때까지 유지됩니다",
        f"감지 사유: {reason}",
        f"미션: {mission_result['mission_title']}",
        mission_result["instruction"],
        f"유지 진행률 {progress}%  |  재도전 횟수 {mission_result['timeout_count']}",
    ]
    frame = TEXT.draw_lines(frame, lines, (x1 + 24, y1 + 18), size=22, line_gap=30, color=(255, 255, 255))

    bar_x1, bar_y1 = x1 + 24, y2 - 24
    bar_x2 = x2 - 24
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + 12), (82, 82, 82), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (int(bar_x1 + (bar_x2 - bar_x1) * mission_result["progress"]), bar_y1 + 12), (0, 230, 255), -1)
    if message:
        frame = TEXT.draw(frame, message, (x1 + 24, y2 + 8), size=19, color=(0, 230, 255))
    return frame


def draw_toast(frame, message: str, seconds_left: float, color=(0, 180, 0)) -> object:
    if not message or seconds_left <= 0:
        return frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = width // 2 - 330, 34, width // 2 + 330, 96
    _transparent_box(frame, x1, y1, x2, y2, color, 0.72)
    return TEXT.draw(frame, message, (x1 + 22, y1 + 15), size=24, color=(255, 255, 255))


def draw_completion_banner(frame, message: str, seconds_left: float) -> object:
    return draw_toast(frame, message, seconds_left, color=(0, 130, 0))
