import cv2
import numpy as np


class MotionDetector:
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        self.prev_frame = None

    def detect(self, frame, threshold=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_frame = gray

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_motion = float(np.mean(magnitude))
        active_threshold = self.threshold if threshold is None else threshold
        return mean_motion > active_threshold, magnitude


_default_detector = MotionDetector()


def motion_detect(frame, threshold=2.5):
    return _default_detector.detect(frame, threshold)
