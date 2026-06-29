from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp

from config import (
    DRAW_FACE_MESH,
    EYE_CLOSED_EAR_THRESHOLD,
    FACE_AWAY_YAW_RATIO_THRESHOLD,
    HEAD_DOWN_RATIO_THRESHOLD,
    MAX_NUM_FACES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from core.types import FaceInfo
from utils.geometry import bbox_from_points, distance_2d
from utils.model_paths import resolve_model_path


# MediaPipe Face Landmarker 인덱스
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152


class FaceMeshDetector:
    """Face Landmarker로 얼굴 메쉬와 집중도용 얼굴 지표를 반환한다."""

    def __init__(
        self,
        model_path: Path,
        draw: bool = DRAW_FACE_MESH,
        max_num_faces: int = MAX_NUM_FACES,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.draw = draw
        self.last_timestamp_ms = -1
        resolved_path = resolve_model_path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker 모델을 찾지 못했습니다: {resolved_path}\n"
                "프로젝트 루트에서 `python download_models.py`를 한 번 실행하세요."
            )

        base_options = mp.tasks.BaseOptions
        face_landmarker = mp.tasks.vision.FaceLandmarker
        face_options = mp.tasks.vision.FaceLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        options = face_options(
            base_options=base_options(model_asset_path=str(resolved_path)),
            running_mode=running_mode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = face_landmarker.create_from_options(options)

    def detect(self, frame) -> Tuple[object, List[FaceInfo]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = self._next_timestamp_ms()
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        height, width = frame.shape[:2]
        faces: List[FaceInfo] = []
        for landmarks in result.face_landmarks or []:
            face = self._to_face_info(landmarks, width, height)
            faces.append(face)
            if self.draw:
                self._draw_face(frame, face)
        return frame, faces

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _to_face_info(self, landmarks, width: int, height: int) -> FaceInfo:
        normalized = [(float(p.x), float(p.y), float(p.z)) for p in landmarks]
        pixels = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        bbox = bbox_from_points(normalized, width, height)

        left_ear = self._eye_aspect_ratio(pixels, LEFT_EYE)
        right_ear = self._eye_aspect_ratio(pixels, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        eyes_closed = ear < EYE_CLOSED_EAR_THRESHOLD

        left_eye = pixels[LEFT_EYE_OUTER]
        right_eye = pixels[RIGHT_EYE_OUTER]
        nose = pixels[NOSE_TIP]
        eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_distance = max(distance_2d(left_eye, right_eye), 1.0)
        yaw_ratio = (nose[0] - eye_mid_x) / eye_distance
        if yaw_ratio > FACE_AWAY_YAW_RATIO_THRESHOLD:
            direction = "right"
        elif yaw_ratio < -FACE_AWAY_YAW_RATIO_THRESHOLD:
            direction = "left"
        else:
            direction = "forward"

        # 카메라 쪽을 정면으로 볼 때의 코 위치를 기준으로 간단한 고개 숙임 비율을 계산한다.
        top = pixels[FOREHEAD]
        chin = pixels[CHIN]
        face_height = max(distance_2d(top, chin), 1.0)
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        head_down_ratio = (nose[1] - eye_mid_y) / face_height
        head_down = head_down_ratio > HEAD_DOWN_RATIO_THRESHOLD

        return FaceInfo(
            landmarks=normalized,
            pixel_landmarks=pixels,
            bbox=bbox,
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            eyes_closed=eyes_closed,
            direction=direction,
            yaw_ratio=yaw_ratio,
            head_down=head_down,
            head_down_ratio=head_down_ratio,
        )

    @staticmethod
    def _eye_aspect_ratio(points: List[Tuple[int, int]], indices: Tuple[int, ...]) -> float:
        p1, p2, p3, p4, p5, p6 = (points[index] for index in indices)
        vertical = distance_2d(p2, p6) + distance_2d(p3, p5)
        horizontal = max(2.0 * distance_2d(p1, p4), 1.0)
        return vertical / horizontal

    def _draw_face(self, frame, face: FaceInfo) -> None:
        x1, y1, x2, y2 = face.bbox
        if self.draw:
            for x, y in face.pixel_landmarks:
                cv2.circle(frame, (x, y), 1, (0, 220, 255), -1)

        color = (0, 255, 0) if not face.eyes_closed else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        status = f"Face | {face.direction} | EAR:{face.ear:.2f}"
        if face.eyes_closed:
            status += " | eyes closed"
        if face.head_down:
            status += " | head down"
        cv2.putText(
            frame,
            status,
            (x1, max(22, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    def close(self) -> None:
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
