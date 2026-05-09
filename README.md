# AI Camera Surveillance

A lightweight intelligent video surveillance project using YOLOv8, DeepSORT, optical flow motion detection, and anomaly detection.

## Project structure

- `main.py` - entrypoint for running the surveillance app
- `camera_stream.py` - camera capture helper
- `detector.py` - YOLO person detector
- `tracker.py` - DeepSORT tracker
- `motion.py` - optical flow motion detection
- `anomaly.py` - autoencoder-based anomaly scoring
- `alerts.py` - email alert sender
- `database.py` - SQLite event logger
- `models/yolov8n.pt` - YOLO model weights
- `logs/` - runtime event storage

## Setup

1. Install required packages:

```powershell
pip install -r requirements.txt
```

2. Run the application:

```powershell
python main.py
```

## Notes

- `logs/` must exist so `logs/events.db` can be created.
- `alerts.py` currently uses hardcoded email credentials; replace them with environment-safe values before production.
