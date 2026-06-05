import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandInfo:
    handedness: str
    finger_count: int
    gesture: str
    landmarks: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]
    finger_states: Dict[str, bool]


class HandDetector:
    """
    MediaPipe Tasks HandLandmarker 기반 손 감지 모듈.

    담당 기능:
    - 손 랜드마크 감지
    - 손 bbox 계산
    - 손가락 개수 추정
    - 간단한 제스처 추정
    - 화면에 손 랜드마크 그리기
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        self.model_path = Path("hand_landmarker.task")
        self._download_model_if_needed()

        self.mp_tasks = mp.tasks
        self.mp_vision = mp.tasks.vision

        base_options = self.mp_tasks.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = self.mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.detector = self.mp_vision.HandLandmarker.create_from_options(options)

    def _download_model_if_needed(self):
        """
        MediaPipe HandLandmarker 모델 파일이 없으면 자동 다운로드.
        """
        if self.model_path.exists():
            return

        print("hand_landmarker.task 모델 파일이 없어 다운로드합니다...")

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

        urllib.request.urlretrieve(model_url, self.model_path)

        print(f"모델 다운로드 완료: {self.model_path}")

    def detect(self, frame) -> List[HandInfo]:
        """
        OpenCV BGR 프레임을 입력받아 손 정보를 반환.
        """
        image_height, image_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect(mp_image)

        hand_infos = []

        if not result.hand_landmarks:
            return hand_infos

        for index, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = "Unknown"

            if result.handedness and index < len(result.handedness):
                if result.handedness[index]:
                    handedness = result.handedness[index][0].category_name

            landmarks = self._extract_landmarks(
                hand_landmarks=hand_landmarks,
                image_width=image_width,
                image_height=image_height,
            )

            bbox = self._calculate_bbox(
                landmarks=landmarks,
                image_width=image_width,
                image_height=image_height,
            )

            finger_states = self._get_finger_states(landmarks)
            finger_count = sum(1 for is_open in finger_states.values() if is_open)

            gesture = self._classify_gesture(
                finger_count=finger_count,
                finger_states=finger_states,
            )

            hand_infos.append(
                HandInfo(
                    handedness=handedness,
                    finger_count=finger_count,
                    gesture=gesture,
                    landmarks=landmarks,
                    bbox=bbox,
                    finger_states=finger_states,
                )
            )

        return hand_infos

    def draw(self, frame, hand_infos: List[HandInfo]):
        """
        감지된 손 bbox, 랜드마크, 제스처를 화면에 그림.
        """
        annotated = frame.copy()

        for hand_info in hand_infos:
            x1, y1, x2, y2 = hand_info.bbox

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                2,
            )

            label = (
                f"{hand_info.handedness} / "
                f"fingers: {hand_info.finger_count} / "
                f"{hand_info.gesture}"
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
            )

            for point_index, point in enumerate(hand_info.landmarks):
                x, y = point

                cv2.circle(
                    annotated,
                    (x, y),
                    4,
                    (255, 0, 255),
                    -1,
                )

                if point_index in [4, 8, 12, 16, 20]:
                    cv2.putText(
                        annotated,
                        str(point_index),
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )

            self._draw_hand_connections(
                frame=annotated,
                landmarks=hand_info.landmarks,
            )

        return annotated

    def close(self):
        self.detector.close()

    def make_summary(self, hand_infos: List[HandInfo]) -> str:
        """
        로그/LLM 프롬프트에 넣기 좋은 손 정보 요약.
        """
        if not hand_infos:
            return "none"

        parts = []

        for idx, hand_info in enumerate(hand_infos, start=1):
            open_fingers = [
                name
                for name, is_open in hand_info.finger_states.items()
                if is_open
            ]

            open_finger_text = ",".join(open_fingers) if open_fingers else "none"

            parts.append(
                f"hand{idx}: {hand_info.handedness}, "
                f"fingers={hand_info.finger_count}, "
                f"gesture={hand_info.gesture}, "
                f"open={open_finger_text}"
            )

        return " / ".join(parts)

    def make_signature(self, hand_infos: List[HandInfo]):
        """
        손 상태 변화 판단용 signature.

        좌표는 계속 흔들리기 때문에 제외.
        손 개수, 좌우, 손가락 개수, 제스처, 열린 손가락 종류만 비교.
        """
        items = []

        for hand_info in hand_infos:
            open_fingers = tuple(
                name
                for name, is_open in sorted(hand_info.finger_states.items())
                if is_open
            )

            items.append(
                (
                    hand_info.handedness,
                    hand_info.finger_count,
                    hand_info.gesture,
                    open_fingers,
                )
            )

        items.sort()

        return tuple(items)

    def _extract_landmarks(
        self,
        hand_landmarks,
        image_width: int,
        image_height: int,
    ):
        landmarks = []

        for landmark in hand_landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            landmarks.append((x, y))

        return landmarks

    def _calculate_bbox(
        self,
        landmarks: List[Tuple[int, int]],
        image_width: int,
        image_height: int,
        padding: int = 20,
    ):
        x_values = [point[0] for point in landmarks]
        y_values = [point[1] for point in landmarks]

        x1 = max(min(x_values) - padding, 0)
        y1 = max(min(y_values) - padding, 0)
        x2 = min(max(x_values) + padding, image_width - 1)
        y2 = min(max(y_values) + padding, image_height - 1)

        return x1, y1, x2, y2

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _angle(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
        c: Tuple[int, int],
    ) -> float:
        """
        점 a-b-c에서 b를 중심으로 한 각도 계산.
        손가락이 펴져 있으면 각도가 180도에 가까움.
        접혀 있으면 각도가 작아짐.
        """
        ab = (a[0] - b[0], a[1] - b[1])
        cb = (c[0] - b[0], c[1] - b[1])

        dot = ab[0] * cb[0] + ab[1] * cb[1]

        ab_length = math.hypot(ab[0], ab[1])
        cb_length = math.hypot(cb[0], cb[1])

        if ab_length == 0 or cb_length == 0:
            return 0.0

        cosine = dot / (ab_length * cb_length)
        cosine = max(-1.0, min(1.0, cosine))

        return math.degrees(math.acos(cosine))

    def _is_finger_open(
        self,
        landmarks: List[Tuple[int, int]],
        mcp_idx: int,
        pip_idx: int,
        dip_idx: int,
        tip_idx: int,
    ) -> bool:
        """
        검지/중지/약지/새끼 손가락 펴짐 판단.

        기존 방식:
        - tip_y < pip_y 만 봄

        개선 방식:
        - PIP 관절 각도
        - DIP 관절 각도
        - 손끝이 손목에서 충분히 멀리 있는지
        를 함께 사용함.
        """
        wrist = landmarks[0]

        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        dip = landmarks[dip_idx]
        tip = landmarks[tip_idx]

        pip_angle = self._angle(mcp, pip, dip)
        dip_angle = self._angle(pip, dip, tip)

        tip_to_wrist = self._distance(tip, wrist)
        pip_to_wrist = self._distance(pip, wrist)

        is_straight = pip_angle > 145 and dip_angle > 150
        is_tip_far = tip_to_wrist > pip_to_wrist * 1.12

        return is_straight and is_tip_far

    def _is_thumb_open(self, landmarks: List[Tuple[int, int]]) -> bool:
        """
        엄지 펴짐 판단.

        엄지는 다른 손가락처럼 위아래 y좌표로 판단하면 잘 틀림.
        그래서 각도와 손바닥 기준 거리로 판단함.
        """
        wrist = landmarks[0]

        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]

        index_mcp = landmarks[5]

        mcp_angle = self._angle(thumb_cmc, thumb_mcp, thumb_ip)
        ip_angle = self._angle(thumb_mcp, thumb_ip, thumb_tip)

        tip_to_wrist = self._distance(thumb_tip, wrist)
        ip_to_wrist = self._distance(thumb_ip, wrist)

        tip_to_index_mcp = self._distance(thumb_tip, index_mcp)
        ip_to_index_mcp = self._distance(thumb_ip, index_mcp)

        is_straight = mcp_angle > 130 and ip_angle > 140
        is_away_from_palm = (
            tip_to_wrist > ip_to_wrist * 1.05
            and tip_to_index_mcp > ip_to_index_mcp * 1.15
        )

        return is_straight and is_away_from_palm

    def _get_finger_states(self, landmarks: List[Tuple[int, int]]) -> Dict[str, bool]:
        """
        각 손가락이 펴졌는지 판단.
        """
        if len(landmarks) < 21:
            return {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            }

        finger_states = {
            "thumb": self._is_thumb_open(landmarks),

            # index: mcp=5, pip=6, dip=7, tip=8
            "index": self._is_finger_open(
                landmarks=landmarks,
                mcp_idx=5,
                pip_idx=6,
                dip_idx=7,
                tip_idx=8,
            ),

            # middle: mcp=9, pip=10, dip=11, tip=12
            "middle": self._is_finger_open(
                landmarks=landmarks,
                mcp_idx=9,
                pip_idx=10,
                dip_idx=11,
                tip_idx=12,
            ),

            # ring: mcp=13, pip=14, dip=15, tip=16
            "ring": self._is_finger_open(
                landmarks=landmarks,
                mcp_idx=13,
                pip_idx=14,
                dip_idx=15,
                tip_idx=16,
            ),

            # pinky: mcp=17, pip=18, dip=19, tip=20
            "pinky": self._is_finger_open(
                landmarks=landmarks,
                mcp_idx=17,
                pip_idx=18,
                dip_idx=19,
                tip_idx=20,
            ),
        }

        return finger_states

    def _classify_gesture(
        self,
        finger_count: int,
        finger_states: Dict[str, bool],
    ) -> str:
        """
        간단한 제스처 분류.
        """
        thumb = finger_states.get("thumb", False)
        index = finger_states.get("index", False)
        middle = finger_states.get("middle", False)
        ring = finger_states.get("ring", False)
        pinky = finger_states.get("pinky", False)

        if finger_count == 0:
            return "fist"

        if finger_count == 5:
            return "open_palm"

        if index and not middle and not ring and not pinky:
            if thumb:
                return "pointing_with_thumb"
            return "pointing"

        if index and middle and not ring and not pinky:
            return "two_fingers"

        if thumb and index and middle and not ring and not pinky:
            return "three_fingers"

        return "hand_detected"

    def _draw_hand_connections(
        self,
        frame,
        landmarks: List[Tuple[int, int]],
    ):
        """
        MediaPipe 손 연결선 직접 그리기.
        """
        connections = [
            # 엄지
            (0, 1), (1, 2), (2, 3), (3, 4),

            # 검지
            (0, 5), (5, 6), (6, 7), (7, 8),

            # 중지
            (0, 9), (9, 10), (10, 11), (11, 12),

            # 약지
            (0, 13), (13, 14), (14, 15), (15, 16),

            # 새끼
            (0, 17), (17, 18), (18, 19), (19, 20),

            # 손바닥 연결
            (5, 9), (9, 13), (13, 17),
        ]

        for start_idx, end_idx in connections:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]

            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 0, 255),
                2,
            )