from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from config import FACE_MODEL_PATH, HAND_MODEL_PATH, MODELS_DIR, POSE_MODEL_PATH

# MediaPipe 공식 모델 저장소의 task bundle URLs
MODEL_URLS = {
    FACE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    HAND_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    POSE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 100_000:
        print(f"[이미 있음] {destination.name}")
        return
    print(f"[다운로드] {destination.name}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    temporary.replace(destination)
    print(f"[완료] {destination}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in MODEL_URLS.items():
        download(url, path)
    print("\n모델 준비 완료. 이제 `python main.py`를 실행하세요.")


if __name__ == "__main__":
    main()
