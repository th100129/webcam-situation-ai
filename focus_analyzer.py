import time
from collections import deque
from math import sqrt


class HoldTimer:
    def __init__(self, hold_seconds):
        self.hold_seconds = hold_seconds
        self.true_since = None
        self.active = False

    def update(self, condition, now=None):
        if now is None:
            now = time.time()

        if condition:
            if self.true_since is None:
                self.true_since = now

            duration = now - self.true_since
            self.active = duration >= self.hold_seconds

            return {
                "active": self.active,
                "duration": duration,
            }

        self.true_since = None
        self.active = False

        return {
            "active": False,
            "duration": 0.0,
        }


class FocusBlurAnalyzer:
    def __init__(
        self,
        face_missing_hold_seconds=2.0,
        face_away_hold_seconds=1.5,
        head_down_hold_seconds=1.3,
        eyes_closed_hold_seconds=5.0,
        hand_face_hold_seconds=1.0,
        pose_unstable_hold_seconds=1.2,
        face_away_ratio_threshold=0.08,
        head_down_ratio_threshold=0.36,
        eye_closed_ear_threshold=0.19,
        pose_movement_threshold=0.025,
        hand_face_padding_ratio=0.12,
        pose_history_size=8,
    ):
        self.face_away_ratio_threshold = face_away_ratio_threshold
        self.head_down_ratio_threshold = head_down_ratio_threshold
        self.eye_closed_ear_threshold = eye_closed_ear_threshold
        self.pose_movement_threshold = pose_movement_threshold
        self.hand_face_padding_ratio = hand_face_padding_ratio

        self.face_missing_timer = HoldTimer(face_missing_hold_seconds)
        self.face_away_timer = HoldTimer(face_away_hold_seconds)
        self.head_down_timer = HoldTimer(head_down_hold_seconds)
        self.eyes_closed_timer = HoldTimer(eyes_closed_hold_seconds)
        self.hand_face_timer = HoldTimer(hand_face_hold_seconds)
        self.pose_unstable_timer = HoldTimer(pose_unstable_hold_seconds)

        self.pose_center_history = deque(maxlen=pose_history_size)

    def analyze(self, face_infos, hand_infos, pose_infos, frame_shape):
        now = time.time()

        face_result = self._analyze_face(face_infos, frame_shape)
        hand_face_result = self._analyze_hand_face_contact(
            face_infos=face_infos,
            hand_infos=hand_infos,
            frame_shape=frame_shape,
        )
        pose_result = self._analyze_pose_movement(
            pose_infos=pose_infos,
            frame_shape=frame_shape,
        )

        raw_face_missing = not face_result["face_detected"]
        raw_face_away = face_result["direction"] in ["left", "right"]
        raw_head_down = face_result["head_down"]
        raw_eyes_closed = face_result["eyes_closed"]
        raw_hand_face_contact = hand_face_result["contact"]
        raw_pose_unstable = pose_result["unstable"]

        face_missing_state = self.face_missing_timer.update(raw_face_missing, now)
        face_away_state = self.face_away_timer.update(raw_face_away, now)
        head_down_state = self.head_down_timer.update(raw_head_down, now)
        eyes_closed_state = self.eyes_closed_timer.update(raw_eyes_closed, now)
        hand_face_state = self.hand_face_timer.update(raw_hand_face_contact, now)
        pose_unstable_state = self.pose_unstable_timer.update(raw_pose_unstable, now)

        stable_indicators = {
            "face_missing": face_missing_state["active"],
            "face_away": face_away_state["active"],
            "head_down": head_down_state["active"],
            "eyes_closed": eyes_closed_state["active"],
            "hand_face_contact": hand_face_state["active"],
            "pose_unstable": pose_unstable_state["active"],
        }

        raw_indicators = {
            "face_missing": raw_face_missing,
            "face_away": raw_face_away,
            "head_down": raw_head_down,
            "eyes_closed": raw_eyes_closed,
            "hand_face_contact": raw_hand_face_contact,
            "pose_unstable": raw_pose_unstable,
        }

        score = self._calculate_score(stable_indicators)
        level = self._get_level(score)

        active_reasons = [
            name
            for name, active in stable_indicators.items()
            if active
        ]

        if not active_reasons:
            active_reasons_text = "none"
        else:
            active_reasons_text = ", ".join(active_reasons)

        summary = (
            f"score: {score}, level: {level}, "
            f"direction: {face_result['direction']}, "
            f"eyes: {face_result['eye_state']}, "
            f"reasons: {active_reasons_text}"
        )

        return {
            "score": score,
            "level": level,
            "summary": summary,
            "active_reasons": active_reasons,
            "raw_indicators": raw_indicators,
            "stable_indicators": stable_indicators,
            "timers": {
                "face_missing": face_missing_state,
                "face_away": face_away_state,
                "head_down": head_down_state,
                "eyes_closed": eyes_closed_state,
                "hand_face_contact": hand_face_state,
                "pose_unstable": pose_unstable_state,
            },
            "face": face_result,
            "hand_face_contact": hand_face_result,
            "pose": pose_result,
        }

    def make_signature(self, analysis):
        stable = analysis["stable_indicators"]

        signature = tuple(
            sorted(
                name
                for name, active in stable.items()
                if active
            )
        )

        return (
            analysis["score"],
            analysis["level"],
            signature,
            analysis["face"]["direction"],
            analysis["face"]["eye_state"],
        )

    def _analyze_face(self, face_infos, frame_shape):
        if not face_infos:
            return self._empty_face_result()

        face = face_infos[0]
        points = self._get_landmark_points(face, frame_shape)
        bbox = self._get_bbox(face, points)

        if not points or bbox is None:
            return self._empty_face_result()

        nose_idx = 1
        forehead_idx = 10
        chin_idx = 152
        left_eye_outer_idx = 33
        right_eye_outer_idx = 263

        required = [
            nose_idx,
            forehead_idx,
            chin_idx,
            left_eye_outer_idx,
            right_eye_outer_idx,
        ]

        if len(points) <= max(required):
            result = self._empty_face_result()
            result["face_detected"] = True
            result["direction"] = "unknown"
            return result

        nose = points[nose_idx]
        forehead = points[forehead_idx]
        chin = points[chin_idx]

        face_width = max(bbox["x2"] - bbox["x1"], 1)
        face_center_x = (bbox["x1"] + bbox["x2"]) / 2

        yaw_ratio = (nose[0] - face_center_x) / face_width

        if yaw_ratio > self.face_away_ratio_threshold:
            direction = "right"
        elif yaw_ratio < -self.face_away_ratio_threshold:
            direction = "left"
        else:
            direction = "center"

        left_eye = points[left_eye_outer_idx]
        right_eye = points[right_eye_outer_idx]

        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        vertical_ref = max(abs(chin[1] - forehead[1]), 1)

        head_down_ratio = (nose[1] - eye_center_y) / vertical_ref
        head_down = head_down_ratio >= self.head_down_ratio_threshold

        eye_ear = self._calculate_average_eye_ear(points)

        if eye_ear is None:
            eyes_closed = False
            eye_state = "unknown"
            eye_ear_value = 0.0
        else:
            eye_ear_value = eye_ear
            eyes_closed = eye_ear <= self.eye_closed_ear_threshold
            eye_state = "closed" if eyes_closed else "open"

        return {
            "face_detected": True,
            "direction": direction,
            "yaw_ratio": yaw_ratio,
            "head_down": head_down,
            "head_down_ratio": head_down_ratio,
            "eyes_closed": eyes_closed,
            "eye_state": eye_state,
            "eye_ear": eye_ear_value,
        }

    def _empty_face_result(self):
        return {
            "face_detected": False,
            "direction": "missing",
            "yaw_ratio": 0.0,
            "head_down": False,
            "head_down_ratio": 0.0,
            "eyes_closed": False,
            "eye_state": "unknown",
            "eye_ear": 0.0,
        }

    def _calculate_average_eye_ear(self, points):
        left_indices = {
            "outer": 33,
            "inner": 133,
            "top1": 160,
            "bottom1": 144,
            "top2": 158,
            "bottom2": 153,
        }

        right_indices = {
            "outer": 362,
            "inner": 263,
            "top1": 385,
            "bottom1": 380,
            "top2": 387,
            "bottom2": 373,
        }

        max_idx = max(list(left_indices.values()) + list(right_indices.values()))

        if len(points) <= max_idx:
            return None

        left_ear = self._calculate_eye_ear(points, left_indices)
        right_ear = self._calculate_eye_ear(points, right_indices)

        return (left_ear + right_ear) / 2

    def _calculate_eye_ear(self, points, indices):
        outer = points[indices["outer"]]
        inner = points[indices["inner"]]
        top1 = points[indices["top1"]]
        bottom1 = points[indices["bottom1"]]
        top2 = points[indices["top2"]]
        bottom2 = points[indices["bottom2"]]

        vertical_1 = self._distance(top1, bottom1)
        vertical_2 = self._distance(top2, bottom2)
        horizontal = max(self._distance(outer, inner), 1)

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _distance(self, p1, p2):
        return sqrt(
            (p1[0] - p2[0]) ** 2
            + (p1[1] - p2[1]) ** 2
        )

    def _analyze_hand_face_contact(self, face_infos, hand_infos, frame_shape):
        if not face_infos or not hand_infos:
            return {
                "contact": False,
                "contact_points": 0,
            }

        face = face_infos[0]
        face_points = self._get_landmark_points(face, frame_shape)
        face_bbox = self._get_bbox(face, face_points)

        if face_bbox is None:
            return {
                "contact": False,
                "contact_points": 0,
            }

        face_width = max(face_bbox["x2"] - face_bbox["x1"], 1)
        face_height = max(face_bbox["y2"] - face_bbox["y1"], 1)

        padding = int(max(face_width, face_height) * self.hand_face_padding_ratio)

        x1 = face_bbox["x1"] - padding
        y1 = face_bbox["y1"] - padding
        x2 = face_bbox["x2"] + padding
        y2 = face_bbox["y2"] + padding

        contact_points = 0
        fingertip_indices = [4, 8, 12, 16, 20]

        for hand in hand_infos:
            hand_points = self._get_landmark_points(hand, frame_shape)

            if not hand_points:
                continue

            candidate_points = []

            for idx in fingertip_indices:
                if idx < len(hand_points):
                    candidate_points.append(hand_points[idx])

            if not candidate_points:
                candidate_points = hand_points

            for point in candidate_points:
                px = point[0]
                py = point[1]

                if x1 <= px <= x2 and y1 <= py <= y2:
                    contact_points += 1

        return {
            "contact": contact_points > 0,
            "contact_points": contact_points,
        }

    def _analyze_pose_movement(self, pose_infos, frame_shape):
        if not pose_infos:
            return {
                "pose_detected": False,
                "unstable": False,
                "movement": 0.0,
            }

        height, width = frame_shape[:2]

        pose = pose_infos[0]
        points = self._get_landmark_points(pose, frame_shape)

        if not points:
            return {
                "pose_detected": False,
                "unstable": False,
                "movement": 0.0,
            }

        center = self._get_pose_center(points)

        if center is None:
            return {
                "pose_detected": True,
                "unstable": False,
                "movement": 0.0,
            }

        normalized_center = (
            center[0] / max(width, 1),
            center[1] / max(height, 1),
        )

        movement = 0.0

        if len(self.pose_center_history) >= 3:
            avg_x = sum(p[0] for p in self.pose_center_history) / len(self.pose_center_history)
            avg_y = sum(p[1] for p in self.pose_center_history) / len(self.pose_center_history)

            movement = sqrt(
                (normalized_center[0] - avg_x) ** 2
                + (normalized_center[1] - avg_y) ** 2
            )

        self.pose_center_history.append(normalized_center)

        unstable = movement >= self.pose_movement_threshold

        return {
            "pose_detected": True,
            "unstable": unstable,
            "movement": movement,
        }

    def _get_pose_center(self, points):
        candidate_indices = [11, 12, 23, 24]
        candidates = []

        for idx in candidate_indices:
            if idx >= len(points):
                continue

            point = points[idx]

            if len(point) >= 4:
                visibility = point[3]
                if visibility < 0.3:
                    continue

            candidates.append(point)

        if not candidates:
            return None

        x = sum(p[0] for p in candidates) / len(candidates)
        y = sum(p[1] for p in candidates) / len(candidates)

        return x, y

    def _calculate_score(self, stable_indicators):
        score = 0

        if stable_indicators["face_missing"]:
            score += 35

        if stable_indicators["face_away"]:
            score += 20

        if stable_indicators["head_down"]:
            score += 25

        if stable_indicators["eyes_closed"]:
            score += 45

        if stable_indicators["hand_face_contact"]:
            score += 20

        if stable_indicators["pose_unstable"]:
            score += 15

        return min(score, 100)

    def _get_level(self, score):
        if score >= 75:
            return "high"
        if score >= 50:
            return "warning"
        if score >= 25:
            return "caution"

        return "normal"

    def _get_bbox(self, item, points):
        bbox = self._get_value(item, ["bbox"], None)

        if bbox is not None:
            return bbox

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        return {
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys),
        }

    def _get_landmark_points(self, item, frame_shape):
        raw_landmarks = self._get_value(
            item,
            [
                "landmarks",
                "points",
                "landmark_points",
                "hand_landmarks",
                "pose_landmarks",
                "face_landmarks",
            ],
            None,
        )

        if raw_landmarks is None:
            return []

        return self._convert_landmarks_to_points(raw_landmarks, frame_shape)

    def _convert_landmarks_to_points(self, raw_landmarks, frame_shape):
        height, width = frame_shape[:2]

        if hasattr(raw_landmarks, "landmark"):
            landmarks = raw_landmarks.landmark
        else:
            landmarks = raw_landmarks

        points = []

        for landmark in landmarks:
            x = None
            y = None
            z = 0.0
            visibility = 1.0

            if isinstance(landmark, dict):
                x = landmark.get("x")
                y = landmark.get("y")
                z = landmark.get("z", 0.0)
                visibility = landmark.get("visibility", 1.0)

            elif isinstance(landmark, (list, tuple)):
                if len(landmark) >= 2:
                    x = landmark[0]
                    y = landmark[1]

                if len(landmark) >= 3:
                    z = landmark[2]

                if len(landmark) >= 4:
                    visibility = landmark[3]

            else:
                x = getattr(landmark, "x", None)
                y = getattr(landmark, "y", None)
                z = getattr(landmark, "z", 0.0)
                visibility = getattr(landmark, "visibility", 1.0)

            if x is None or y is None:
                continue

            if visibility is None:
                visibility = 1.0

            if 0 <= x <= 2 and 0 <= y <= 2:
                px = int(x * width)
                py = int(y * height)
            else:
                px = int(x)
                py = int(y)

            points.append((px, py, z, visibility))

        return points

    def _get_value(self, item, names, default=None):
        for name in names:
            if isinstance(item, dict) and name in item:
                return item[name]

            if hasattr(item, name):
                return getattr(item, name)

        return default