import time

import cv2
import mediapipe as mp


class FaceMeshDetector:
    def __init__(
        self,
        model_path="face_landmarker.task",
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
        draw=True
    ):
        self.draw = draw
        self.timestamp_ms = 0
        self.last_timestamp_ms = 0

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        웹캠 프레임에서 얼굴 랜드마크를 감지한다.

        return:
            frame: 시각화가 적용된 프레임
            faces: 감지된 얼굴 정보 리스트
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        current_timestamp_ms = int(time.time() * 1000)

        if current_timestamp_ms <= self.last_timestamp_ms:
            current_timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = current_timestamp_ms
        self.timestamp_ms = current_timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        faces = []

        if result.face_landmarks:
            height, width, _ = frame.shape

            for face_idx, face_landmarks in enumerate(result.face_landmarks):
                points = []

                for landmark in face_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    z = landmark.z
                    points.append((x, y, z))

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                bbox = {
                    "x1": min(xs),
                    "y1": min(ys),
                    "x2": max(xs),
                    "y2": max(ys)
                }

                faces.append({
                    "face_id": face_idx,
                    "bbox": bbox,
                    "landmarks": points
                })

                if self.draw:
                    self._draw_face_mesh(
                        frame=frame,
                        points=points,
                        bbox=bbox,
                        face_idx=face_idx
                    )

        return frame, faces

    def _draw_face_mesh(self, frame, points, bbox, face_idx):
        # 전체 랜드마크 점 표시
        for x, y, _ in points:
            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 255),
                -1
            )

        # 얼굴 bbox
        cv2.rectangle(
            frame,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Face {face_idx + 1}",
            (bbox["x1"], max(bbox["y1"] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # 얼굴 윤곽
        self._draw_polyline(
            frame,
            points,
            [
                10, 338, 297, 332, 284, 251, 389, 356,
                454, 323, 361, 288, 397, 365, 379, 378,
                400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21,
                54, 103, 67, 109, 10
            ],
            color=(255, 255, 0),
            thickness=1
        )

        # 왼쪽 눈
        self._draw_polyline(
            frame,
            points,
            [33, 246, 161, 160, 159, 158, 157, 173, 133],
            color=(255, 0, 255),
            thickness=2
        )
        self._draw_polyline(
            frame,
            points,
            [33, 7, 163, 144, 145, 153, 154, 155, 133],
            color=(255, 0, 255),
            thickness=2
        )

        # 오른쪽 눈
        self._draw_polyline(
            frame,
            points,
            [362, 398, 384, 385, 386, 387, 388, 466, 263],
            color=(255, 0, 255),
            thickness=2
        )
        self._draw_polyline(
            frame,
            points,
            [362, 382, 381, 380, 374, 373, 390, 249, 263],
            color=(255, 0, 255),
            thickness=2
        )

        # 입
        self._draw_polyline(
            frame,
            points,
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
            color=(255, 0, 255),
            thickness=2
        )
        self._draw_polyline(
            frame,
            points,
            [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            color=(255, 0, 255),
            thickness=2
        )

        # 코
        self._draw_polyline(
            frame,
            points,
            [168, 6, 197, 195, 5, 4, 1],
            color=(255, 0, 255),
            thickness=2
        )

    def _draw_polyline(self, frame, points, indices, color=(255, 255, 0), thickness=1):
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]

            if start_idx >= len(points) or end_idx >= len(points):
                continue

            x1, y1, _ = points[start_idx]
            x2, y2, _ = points[end_idx]

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

    def close(self):
        self.landmarker.close()