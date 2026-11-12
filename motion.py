import cv2
import numpy as np

prev_frame = None


def motion_detect(frame, threshold=2.5):
    global prev_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray
        return False, None

    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Calculate motion magnitude
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_motion = np.mean(magnitude)

    prev_frame = gray
    return mean_motion > threshold, magnitude