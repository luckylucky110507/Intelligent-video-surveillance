import os

import cv2


def get_stream(source=0):
    if os.name == "nt" and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open camera stream: {source!r}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap