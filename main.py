import cv2, threading
from camera_stream import get_stream
from detector      import detect, draw_boxes
from tracker       import track_persons
from motion        import dl_motion_detect
from anomaly       import detect_anomaly
from alerts        import send_alert
from database      import init_db, log_event


def main():
    init_db()
    cap = get_stream(source=0)
    frame_count = 0
    output_frame = None
    lock = threading.Lock()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        motion, _ = dl_motion_detect(frame)

        if motion:
            # Step 1: Detect persons (YOLOv8)
            results = detect(frame)
            frame = draw_boxes(frame, results)

            # Step 2: Track persons (DeepSORT)
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 0))
            frame = track_persons(frame, detections)

            # Step 3: Anomaly detection (CNN Autoencoder)
            is_anomaly, score = detect_anomaly(frame)
            if is_anomaly:
                send_alert("Anomaly detected", score=score)
                log_event("ANOMALY", score=score)
                cv2.putText(frame, f"ANOMALY! score={score}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Step 4: Crowd alert
            if len(results.boxes) > 3:
                send_alert("Crowd detected", score=len(results.boxes))
                log_event("CROWD", score=len(results.boxes))

        with lock:
            output_frame = frame.copy()

        cv2.imshow("Intelligent Video Surveillance", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()