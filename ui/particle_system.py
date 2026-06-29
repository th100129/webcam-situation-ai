from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class _Particle:
    """축하 효과에만 사용하는 화면 좌표 기반 파티클."""

    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: Tuple[int, int, int]
    kind: str
    rotation: float
    rotation_speed: float


class ParticleSystem:
    """미션 성공 순간에 색종이 조각과 반짝임이 터지는 파티클 효과."""

    COLORS: tuple[Tuple[int, int, int], ...] = (
        (0, 230, 255),     # 노랑
        (80, 255, 80),     # 초록
        (255, 120, 255),   # 분홍
        (255, 210, 40),    # 하늘색
        (255, 255, 255),   # 흰색
        (0, 165, 255),     # 주황
    )

    def __init__(self) -> None:
        self.particles: List[_Particle] = []

    @property
    def active(self) -> bool:
        return bool(self.particles)

    def celebrate(self, center: Tuple[int, int], frame_size: Tuple[int, int]) -> None:
        """미션 성공 시 호출한다.

        중심에서는 위로 퍼지는 색종이 조각을, 화면 상단에서는 가볍게 떨어지는
        반짝이를 만들기 때문에 한 번의 성공에도 화면 전체가 축하 효과처럼 보인다.
        """
        frame_width, frame_height = frame_size
        center_x, center_y = center

        # 얼굴 주변에서 위로 한 번 크게 터지는 메인 버스트
        self.burst((center_x, center_y), count=62, speed_scale=1.0)

        # 상단 양쪽에서 떨어지는 작은 색종이 조각
        self._rain_confetti(frame_width, frame_height, count=26)

    def burst(self, center: Tuple[int, int], count: int = 58, speed_scale: float = 1.0) -> None:
        """위로 솟았다가 떨어지는 색종이/반짝이 묶음을 만든다."""
        x, y = center
        for _ in range(count):
            speed = random.uniform(145.0, 325.0) * speed_scale
            # 위쪽 반원으로 터뜨려 얼굴을 너무 오래 가리지 않게 한다.
            angle = random.uniform(math.radians(202), math.radians(338))
            life = random.uniform(0.85, 1.55)
            kind = random.choices(("confetti", "spark", "dot"), weights=(56, 28, 16), k=1)[0]
            self.particles.append(
                _Particle(
                    x=float(x + random.uniform(-12, 12)),
                    y=float(y + random.uniform(-12, 12)),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=life,
                    max_life=life,
                    size=random.uniform(5.0, 10.0),
                    color=random.choice(self.COLORS),
                    kind=kind,
                    rotation=random.uniform(0.0, math.tau),
                    rotation_speed=random.uniform(-8.0, 8.0),
                )
            )

    def _rain_confetti(self, frame_width: int, frame_height: int, count: int) -> None:
        for _ in range(count):
            life = random.uniform(0.95, 1.75)
            self.particles.append(
                _Particle(
                    x=random.uniform(frame_width * 0.16, frame_width * 0.84),
                    y=random.uniform(-frame_height * 0.08, frame_height * 0.08),
                    vx=random.uniform(-70.0, 70.0),
                    vy=random.uniform(45.0, 135.0),
                    life=life,
                    max_life=life,
                    size=random.uniform(4.0, 9.0),
                    color=random.choice(self.COLORS),
                    kind=random.choice(("confetti", "spark")),
                    rotation=random.uniform(0.0, math.tau),
                    rotation_speed=random.uniform(-7.0, 7.0),
                )
            )

    @staticmethod
    def _fade_color(color: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
        # 효과가 끝날 때 갑자기 사라지지 않게 밝기를 서서히 낮춘다.
        brightness = 0.22 + 0.78 * alpha
        return tuple(max(0, min(255, int(channel * brightness))) for channel in color)

    def _draw_confetti(self, frame, particle: _Particle, alpha: float) -> None:
        half_width = max(2.0, particle.size * (0.65 + alpha * 0.55))
        half_height = max(1.5, particle.size * 0.34)
        cosine = math.cos(particle.rotation)
        sine = math.sin(particle.rotation)
        local_corners = ((-half_width, -half_height), (half_width, -half_height), (half_width, half_height), (-half_width, half_height))
        points = []
        for local_x, local_y in local_corners:
            points.append(
                (
                    int(particle.x + local_x * cosine - local_y * sine),
                    int(particle.y + local_x * sine + local_y * cosine),
                )
            )
        cv2.fillConvexPoly(frame, np.asarray(points, dtype=np.int32), self._fade_color(particle.color, alpha), lineType=cv2.LINE_AA)

    def _draw_spark(self, frame, particle: _Particle, alpha: float) -> None:
        size = max(3, int(particle.size * (0.55 + alpha)))
        color = self._fade_color(particle.color, alpha)
        center = (int(particle.x), int(particle.y))
        cv2.drawMarker(frame, center, color, markerType=cv2.MARKER_STAR, markerSize=size * 2, thickness=1, line_type=cv2.LINE_AA)
        cv2.circle(frame, center, max(1, size // 4), color, -1, cv2.LINE_AA)

    def _draw_dot(self, frame, particle: _Particle, alpha: float) -> None:
        radius = max(1, int(particle.size * alpha))
        color = self._fade_color(particle.color, alpha)
        previous = (int(particle.x - particle.vx * 0.025), int(particle.y - particle.vy * 0.025))
        current = (int(particle.x), int(particle.y))
        cv2.line(frame, previous, current, color, max(1, radius // 2), cv2.LINE_AA)
        cv2.circle(frame, current, radius, color, -1, cv2.LINE_AA)

    def update_and_draw(self, frame, delta_seconds: float) -> None:
        """생존 중인 파티클을 갱신하고 현재 프레임 위에 그린다."""
        delta_seconds = min(max(delta_seconds, 0.0), 0.12)
        alive: List[_Particle] = []

        for particle in self.particles:
            particle.life -= delta_seconds
            if particle.life <= 0:
                continue

            # 중력, 공기 저항, 회전을 적용한다.
            particle.x += particle.vx * delta_seconds
            particle.y += particle.vy * delta_seconds
            particle.vx *= 0.987
            particle.vy += 440.0 * delta_seconds
            particle.rotation += particle.rotation_speed * delta_seconds

            alpha = particle.life / particle.max_life
            if particle.kind == "confetti":
                self._draw_confetti(frame, particle, alpha)
            elif particle.kind == "spark":
                self._draw_spark(frame, particle, alpha)
            else:
                self._draw_dot(frame, particle, alpha)
            alive.append(particle)

        self.particles = alive
