from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float, float]
BBox = Tuple[int, int, int, int]


@dataclass
class HandInfo:
    handedness: str
    landmarks: List[Point]
    pixel_landmarks: List[Tuple[int, int]]
    finger_states: List[bool]
    finger_count: int
    gesture: str
    bbox: BBox


@dataclass
class FaceInfo:
    landmarks: List[Point]
    pixel_landmarks: List[Tuple[int, int]]
    bbox: BBox
    left_ear: float
    right_ear: float
    ear: float
    eyes_closed: bool
    direction: str
    yaw_ratio: float
    head_down: bool
    head_down_ratio: float


@dataclass
class PoseInfo:
    landmarks: List[Point]
    pixel_landmarks: List[Tuple[int, int]]
    bbox: BBox
    shoulder_center: Tuple[float, float]
    shoulder_width: float
    shoulder_tilt: float
    motion: float
    unstable: bool
    hands_up: bool


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    radius: float
    color: Tuple[int, int, int]
