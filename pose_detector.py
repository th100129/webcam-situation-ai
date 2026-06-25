# pose_detector.py

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),

    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),

    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),

    (11, 23), (12, 24),
    (23, 24),

    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (27, 31),

    (24, 26), (26, 28),
    (28, 30), (30, 32),
    (28, 32),
]


class PoseInfoList(list):
    """
    list처럼 쓰면서도 pose_infos.get("detected") 같은 코드가 있어도 터지지 않게 하는 호환용 클래스
    """

    def get(self, key, default=None):
        if key in ["detected", "pose_detected"]:
            return len(self) > 0

        if len(self) == 0:
            return default

        first_pose = self[0]

        if isinstance(first_pose, dict):
            return first_pose.get(key, default)

        return default


class PoseDetector:
    def __init__(
        self,
        model_path="pose_landmarker_lite.task",
        max_num_poses=1,
        min_detection_confidence=0.5,
        min_pose_detection_confidence=None,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        draw=True,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        **kwargs,
    ):
        self.default_draw = draw
        self.max_num_poses = max_num_poses

        if min_pose_detection_confidence is None:
            min_pose_detection_confidence = min_detection_confidence

        self.model_path = Path(model_path)

        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parent / self.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pose 모델 파일을 찾을 수 없습니다: {self.model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=max_num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.result = None
        self.pose_infos = PoseInfoList()
        self.landmarks = []

        print("MediaPipe Pose 로딩 중 ...")

    def detect(self, frame, draw=None):
        """
        main.py 호환용 반환 형식:
            annotated_frame, pose_infos = pose_detector.detect(frame)

        pose_infos는 list 형태:
            [
                {
                    "detected": True,
                    "pose_detected": True,
                    "landmarks": [...],
                    "pose_landmarks": [...],
                    "result": ...
                }
            ]
        """

        if frame is None:
            self.pose_infos = PoseInfoList()
            self.landmarks = []
            return frame, self.pose_infos

        if draw is None:
            draw = self.default_draw

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        self.result = self.detector.detect(mp_image)

        self.pose_infos = PoseInfoList()
        self.landmarks = []

        detected = (
            self.result is not None
            and self.result.pose_landmarks is not None
            and len(self.result.pose_landmarks) > 0
        )

        if not detected:
            return frame, self.pose_infos

        h, w = frame.shape[:2]

        for pose_idx, pose_landmarks in enumerate(self.result.pose_landmarks):
            landmarks = []

            for landmark_id, lm in enumerate(pose_landmarks):
                visibility = getattr(lm, "visibility", 1.0)
                presence = getattr(lm, "presence", 1.0)

                landmark_info = {
                    "id": landmark_id,
                    "x": int(lm.x * w),
                    "y": int(lm.y * h),
                    "z": lm.z,
                    "visibility": visibility,
                    "presence": presence,
                    "x_norm": lm.x,
                    "y_norm": lm.y,
                }

                landmarks.append(landmark_info)

            pose_info = {
                "id": pose_idx,
                "detected": True,
                "pose_detected": True,
                "landmarks": landmarks,
                "pose_landmarks": landmarks,
                "result": self.result,
            }

            self.pose_infos.append(pose_info)

        if len(self.pose_infos) > 0:
            self.landmarks = self.pose_infos[0]["landmarks"]

        if draw:
            for pose_info in self.pose_infos:
                self.draw_landmarks(frame, pose_info["landmarks"])

        return frame, self.pose_infos

    def find_pose(self, frame, draw=None):
        """
        예전 코드 호환용
        """
        return self.detect(frame, draw=draw)

    def draw_landmarks(self, frame, landmarks):
        if frame is None or not landmarks:
            return frame

        points = {}

        for lm in landmarks:
            visibility = lm.get("visibility", 1.0)

            if visibility < 0.3:
                continue

            landmark_id = lm["id"]
            x = lm["x"]
            y = lm["y"]

            points[landmark_id] = (x, y)

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for start, end in POSE_CONNECTIONS:
            if start in points and end in points:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 255),
                    2,
                )

        return frame

    def get_landmark(self, landmark_id):
        """
        첫 번째 사람의 특정 포즈 랜드마크 가져오기
        """
        if not self.landmarks:
            return None

        for lm in self.landmarks:
            if lm["id"] == landmark_id:
                return lm

        return None

    def get_landmarks(self):
        """
        첫 번째 사람의 전체 랜드마크 반환
        """
        return self.landmarks

    def get_pose_infos(self):
        """
        전체 pose_infos 반환
        """
        return self.pose_infos

    def close(self):
        if self.detector:
            self.detector.close()