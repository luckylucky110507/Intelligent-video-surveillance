from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")  # nano = fastest; use yolov8m.pt for better accuracy

def detect(frame):
    results = model(frame, conf=0.4, classes=[0])  # class 0 = person
    return results[0]  # bounding boxes, confidence scores, class IDs

# Draw boxes
def draw_boxes(frame, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame