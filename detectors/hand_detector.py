from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp

from config import DRAW_HAND_LANDMARKS, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from core.types import HandInfo
from utils.geometry import bbox_from_points, distance_2d
from utils.model_paths import resolve_model_path


WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18
MIDDLE_MCP = 9


class HandDetector:
    """Hand Landmarker 결과를 손가락 수와 제스처로 정리한다."""

    def __init__(
        self,
        model_path: Path,
        draw: bool = DRAW_HAND_LANDMARKS,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.draw = draw
        self.last_timestamp_ms = -1
        resolved_path = resolve_model_path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Hand Landmarker 모델을 찾지 못했습니다: {resolved_path}\n"
                "프로젝트 루트에서 `python download_models.py`를 한 번 실행하세요."
            )

        base_options = mp.tasks.BaseOptions
        hand_landmarker = mp.tasks.vision.HandLandmarker
        hand_options = mp.tasks.vision.HandLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode
        options = hand_options(
            base_options=base_options(model_asset_path=str(resolved_path)),
            running_mode=running_mode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = hand_landmarker.create_from_options(options)

    def detect(self, frame) -> Tuple[object, List[HandInfo]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
        height, width = frame.shape[:2]
        hands: List[HandInfo] = []

        for index, landmarks in enumerate(result.hand_landmarks or []):
            handedness = "Unknown"
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                handedness = result.handedness[index][0].category_name
            hand = self._to_hand_info(landmarks, handedness, width, height)
            hands.append(hand)
            if self.draw:
                self._draw_hand(frame, hand)
        return frame, hands

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _to_hand_info(self, landmarks, handedness: str, width: int, height: int) -> HandInfo:
        normalized = [(float(p.x), float(p.y), float(p.z)) for p in landmarks]
        pixels = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        bbox = bbox_from_points(normalized, width, height)
        states = self._finger_states(normalized)
        finger_count = sum(states)
        gesture = self._classify_gesture(states, normalized)
        return HandInfo(
            handedness=handedness,
            landmarks=normalized,
            pixel_landmarks=pixels,
            finger_states=states,
            finger_count=finger_count,
            gesture=gesture,
            bbox=bbox,
        )

    @staticmethod
    def _finger_states(points: Sequence[Sequence[float]]) -> List[bool]:
        wrist = points[WRIST]
        palm_scale = max(distance_2d(wrist, points[MIDDLE_MCP]), 0.03)

        def is_extended(tip_index: int, pip_index: int, tolerance: float = 0.08) -> bool:
            tip_distance = distance_2d(points[tip_index], wrist)
            pip_distance = distance_2d(points[pip_index], wrist)
            return tip_distance > pip_distance + palm_scale * tolerance

        thumb = is_extended(THUMB_TIP, THUMB_IP, tolerance=0.03)
        index = is_extended(INDEX_TIP, INDEX_PIP)
        middle = is_extended(MIDDLE_TIP, MIDDLE_PIP)
        ring = is_extended(RING_TIP, RING_PIP)
        pinky = is_extended(PINKY_TIP, PINKY_PIP)
        return [thumb, index, middle, ring, pinky]

    @staticmethod
    def _classify_gesture(states: List[bool], points: Sequence[Sequence[float]]) -> str:
        thumb, index, middle, ring, pinky = states
        if all(states):
            return "open_palm"
        if not any(states):
            return "fist"
        if index and middle and not ring and not pinky:
            return "v_sign"
        if index and not middle and not ring and not pinky:
            return "pointing"
        if thumb and not index and not middle and not ring and not pinky:
            # 엄지끝이 손목보다 위에 있을 때만 thumbs_up으로 인정한다.
            if points[THUMB_TIP][1] < points[THUMB_IP][1]:
                return "thumbs_up"
        return "unknown"

    def _draw_hand(self, frame, hand: HandInfo) -> None:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        ]
        for start, end in connections:
            cv2.line(frame, hand.pixel_landmarks[start], hand.pixel_landmarks[end], (255, 160, 0), 2)
        for x, y in hand.pixel_landmarks:
            cv2.circle(frame, (x, y), 3, (0, 230, 255), -1)

        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
        label = f"{hand.handedness} | {hand.gesture} | {hand.finger_count} fingers"
        cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 2)

    def close(self) -> None:
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
