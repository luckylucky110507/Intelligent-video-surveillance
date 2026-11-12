import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


def create_tracker():
    return DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)


tracker = create_tracker()


def track_persons(frame, detections, tracker_instance=None):
    # detections = list of ([x, y, w, h], confidence, class_id)
    active_tracker = tracker_instance or tracker
    tracks = active_tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    return frame