from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import cv2


class ObjectDetector:
    """YOLO 객체 감지를 분리한 모듈. ultralytics가 모델을 자동 다운로드할 수도 있다."""

    def __init__(self, model_path: Path, confidence: float, image_size: int, target_classes: Optional[Set[str]] = None):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("ultralytics가 없습니다. `python -m pip install ultralytics`를 실행하세요.") from error

        self.confidence = confidence
        self.image_size = image_size
        self.target_classes = target_classes
        self.model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")

    def detect(self, frame) -> List[Dict]:
        result = self.model(frame, conf=self.confidence, imgsz=self.image_size, verbose=False)[0]
        detections: List[Dict] = []
        if result.boxes is None:
            return detections

        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names[class_id] if not isinstance(names, dict) else names.get(class_id, str(class_id))
            if self.target_classes is not None and class_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "name": class_name,
                    "confidence": float(box.conf[0].item()),
                    "bbox": (x1, y1, x2, y2),
                }
            )
        return detections

    @staticmethod
    def draw(frame, detections: Iterable[Dict]):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['name']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return frame

    @staticmethod
    def summary(detections: Iterable[Dict]) -> str:
        counts = Counter(det["name"] for det in detections)
        return ", ".join(f"{name}:{count}" for name, count in counts.items()) if counts else "none"
