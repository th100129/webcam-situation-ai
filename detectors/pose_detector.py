from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp

from config import (
    DRAW_POSE_LANDMARKS,
    MAX_NUM_POSES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    POSE_MOTION_THRESHOLD,
    POSE_TILT_THRESHOLD,
)
from core.types import PoseInfo
from utils.geometry import bbox_from_points, distance_2d
from utils.model_paths import resolve_model_path

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16


class PoseDetector:
    """Pose Landmarker로 어깨 기울기, 움직임, 양손 들기를 계산한다."""

    def __init__(
        self,
        model_path: Path,
        draw: bool = DRAW_POSE_LANDMARKS,
        max_num_poses: int = MAX_NUM_POSES,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.draw = draw
        self.last_timestamp_ms = -1
        self.previous_center: Tuple[float, float] | None = None
        resolved_path = resolve_model_path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Pose Landmarker 모델을 찾지 못했습니다: {resolved_path}\n"
                "프로젝트 루트에서 `python download_models.py`를 한 번 실행하세요."
            )

        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        pose_options = mp.tasks.vision.PoseLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode
        options = pose_options(
            base_options=base_options(model_asset_path=str(resolved_path)),
            running_mode=running_mode.VIDEO,
            num_poses=max_num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = pose_landmarker.create_from_options(options)

    def detect(self, frame) -> Tuple[object, List[PoseInfo]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
        height, width = frame.shape[:2]
        poses: List[PoseInfo] = []

        for landmarks in result.pose_landmarks or []:
            pose = self._to_pose_info(landmarks, width, height)
            poses.append(pose)
            if self.draw:
                self._draw_pose(frame, pose)
        if not poses:
            self.previous_center = None
        return frame, poses

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _to_pose_info(self, landmarks, width: int, height: int) -> PoseInfo:
        normalized = [(float(p.x), float(p.y), float(p.z)) for p in landmarks]
        pixels = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        bbox = bbox_from_points(normalized, width, height)
        left_shoulder = normalized[LEFT_SHOULDER]
        right_shoulder = normalized[RIGHT_SHOULDER]
        left_wrist = normalized[LEFT_WRIST]
        right_wrist = normalized[RIGHT_WRIST]

        center = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
        shoulder_width = max(distance_2d(left_shoulder, right_shoulder), 0.02)
        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) / shoulder_width
        if self.previous_center is None:
            motion = 0.0
        else:
            motion = distance_2d(center, self.previous_center) / shoulder_width
        self.previous_center = center

        unstable = shoulder_tilt > POSE_TILT_THRESHOLD or motion > POSE_MOTION_THRESHOLD
        hands_up = left_wrist[1] < center[1] and right_wrist[1] < center[1]

        return PoseInfo(
            landmarks=normalized,
            pixel_landmarks=pixels,
            bbox=bbox,
            shoulder_center=center,
            shoulder_width=shoulder_width,
            shoulder_tilt=shoulder_tilt,
            motion=motion,
            unstable=unstable,
            hands_up=hands_up,
        )

    def _draw_pose(self, frame, pose: PoseInfo) -> None:
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
        ]
        for start, end in connections:
            cv2.line(frame, pose.pixel_landmarks[start], pose.pixel_landmarks[end], (255, 0, 255), 2)
        for index in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            x, y = pose.pixel_landmarks[index]
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

        x1, y1, _, _ = pose.bbox
        label = f"Pose | tilt:{pose.shoulder_tilt:.2f} motion:{pose.motion:.2f}"
        if pose.hands_up:
            label += " | hands up"
        cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    def close(self) -> None:
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
