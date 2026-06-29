from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2


def save_screenshot(frame, output_dir: Path, prefix: str = "capture") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = output_dir / filename
    cv2.imwrite(str(path), frame)
    return path
