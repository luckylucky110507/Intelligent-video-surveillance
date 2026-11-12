import os

import cv2

from alerts import send_alert
from anomaly_rules import detect_rule_anomaly
from camera_stream import get_stream
from database import init_db, log_event
from detector import detect
from motion import MotionDetector
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
    peak_motion_since_detection = 0.0
    motion_detector = MotionDetector()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stream ended or failed.")
                break

            frame_count += 1
            detections = []
            _, motion_magnitude = motion_detector.detect(frame)
            motion_score = (
                float(motion_magnitude.mean()) if motion_magnitude is not None else 0.0
            )
            peak_motion_since_detection = max(
                peak_motion_since_detection, motion_score
            )

            if frame_count % 3 == 0:
                motion_score = peak_motion_since_detection
                peak_motion_since_detection = 0.0
                results = detect(frame)
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    detections.append(
                        ([x1, y1, x2 - x1, y2 - y1], confidence, 0)
                    )

                is_anomaly, reasons = detect_rule_anomaly(
                    len(results.boxes), motion_score
                )
                if is_anomaly:
                    for reason in reasons:
                        if reason == "crowd":
                            event_type = "CROWD"
                            score = len(results.boxes)
                            label = "Crowd detected"
                        else:
                            event_type = "HIGH_MOTION"
                            score = motion_score
                            label = "High motion detected"
                        if send_alert(label, score=score):
                            log_event(event_type, score=score)

                    reason_text = ", ".join(reason.replace("_", " ") for reason in reasons)
                    cv2.putText(
                        frame,
                        f"ANOMALY: {reason_text}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

            frame = track_persons(frame, detections)
            cv2.imshow("Intelligent Video Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
