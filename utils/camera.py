from __future__ import annotations

import os

import cv2


def open_camera(index: int, width: int, height: int, fps: int = 30):
    """가능하면 MJPG 1280x720로 웹캠을 열고, 실패하면 카메라 기본 해상도를 사용한다."""
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return cap

    # MJPG는 Windows 웹캠에서 고해상도 프레임을 안정적으로 받는 데 도움이 되는 경우가 많다.
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def camera_resolution(cap) -> tuple[int, int, float]:
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    )
