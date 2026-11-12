import os

import cv2

from alerts import send_alert
from anomaly import detect_anomaly
from camera_stream import get_stream
from database import init_db, log_event
from detector import detect
from motion import motion_detect
from tracker import track_persons


def main():
    os.makedirs("logs", exist_ok=True)
    init_db()

    try:
        cap = get_stream(source=0)
    except RuntimeError as exc:
        print(exc)
        return

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stream ended or failed.")
                break

            frame_count += 1
            detections = []

            if frame_count % 3 == 0:
                motion, _ = motion_detect(frame)
                if motion:
                    results = detect(frame)
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        detections.append(
                            ([x1, y1, x2 - x1, y2 - y1], confidence, 0)
                        )

                    is_anomaly, score = detect_anomaly(frame)
                    if is_anomaly is True and send_alert("Anomaly detected", score=score):
                        log_event("ANOMALY", score=score)
                        cv2.putText(
                            frame,
                            f"ANOMALY! score={score}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

                    if len(results.boxes) > 3 and send_alert(
                        "Crowd detected", score=len(results.boxes)
                    ):
                        log_event("CROWD", score=len(results.boxes))

            frame = track_persons(frame, detections)
            cv2.imshow("Intelligent Video Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
