from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple


def distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point_in_bbox(point: Sequence[float], bbox: Tuple[int, int, int, int], padding: float = 0.0) -> bool:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x1 -= width * padding
    x2 += width * padding
    y1 -= height * padding
    y2 += height * padding
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def bbox_from_points(points: Iterable[Sequence[float]], frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    pts = list(points)
    if not pts:
        return (0, 0, 0, 0)
    xs = [int(float(p[0]) * frame_width) for p in pts]
    ys = [int(float(p[1]) * frame_height) for p in pts]
    return (max(0, min(xs)), max(0, min(ys)), min(frame_width - 1, max(xs)), min(frame_height - 1, max(ys)))
