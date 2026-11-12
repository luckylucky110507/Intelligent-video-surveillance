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
- `streamlit_app.py` - Streamlit application for uploaded-video analysis

## Run locally

1. Install required packages:

```powershell
pip install -r requirements.txt
```

2. Run the desktop webcam application:

```powershell
python main.py
```

3. Or run the Streamlit app:

```powershell
python -m streamlit run streamlit_app.py
```

The Streamlit app analyzes uploaded video files and provides a downloadable CSV report. It samples video frames to control processing time. A trained `models/autoencoder.pth` is required for meaningful anomaly detection; without it, anomaly scoring is disabled. Local webcam capture is available through `main.py`, not through a remotely hosted Streamlit server.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `models/yolov8n.pt`; do not commit credentials or private video files.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
3. Select the repository and branch, set the main file path to `streamlit_app.py`, then deploy.
4. The app installs dependencies from `requirements.txt`. Video uploads are limited to 100 MB by `.streamlit/config.toml`.

Cloud-hosted apps cannot access a webcam connected to your personal computer. Upload a video in the Streamlit page instead. CPU inference can be slow; increase the frame interval or reduce the video size for faster analysis.

## Notes

- `logs/` is created automatically when the desktop app initializes the database.
- Email alerts are optional. Set `EMAIL_FROM`, `EMAIL_TO`, and `EMAIL_APP_PASS` environment variables before production use; if they are not set, alerts are skipped instead of crashing the app.
- If no camera is connected, the app exits cleanly with a message instead of failing at runtime.
