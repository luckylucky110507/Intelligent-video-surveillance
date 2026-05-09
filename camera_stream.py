import cv2

def get_stream(source=0):  # 0 = webcam, or "rtsp://..." for IP cam
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap