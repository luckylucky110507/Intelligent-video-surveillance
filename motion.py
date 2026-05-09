import cv2
import numpy as np
import torch
import torch.nn as nn

# Deep Learning approach: use dense optical flow (Farneback)
# Better than MOG2 because it uses pixel-level motion vectors

prev_frame = None

def dl_motion_detect(frame, threshold=2.5):
    global prev_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray
        return False, None

    # Farneback optical flow (DL-style dense flow)
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Calculate motion magnitude
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_motion = np.mean(magnitude)

    prev_frame = gray
    return mean_motion > threshold, magnitude