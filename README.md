# Intelligent Video Surveillance

Person detection and tracking for webcam/RTSP streams and uploaded videos, built with YOLOv8, DeepSORT, OpenCV, and Streamlit.

## Features

- Detect people with the YOLOv8n model.
- Gate desktop-camera inference with Farneback optical-flow motion detection.
- Track detections with DeepSORT and display track IDs.
- Analyze uploaded videos in Streamlit and download a per-frame CSV report.
- Optionally log events to SQLite and send email alerts with a 60-second per-event cooldown.

## Architecture

**Desktop camera:** Camera/RTSP → Farneback motion gate → YOLO person detection (every third frame) → DeepSORT tracking (every frame) → experimental anomaly detector → throttled email alerts and SQLite event log.

**Streamlit:** Uploaded video → sampled frames → YOLO person detection → optional DeepSORT tracking → downloadable CSV report.

Farneback is a classical computer-vision optical-flow method; it is not deep learning.

## Run locally

Use Python 3.12 or another Python version supported by the pinned PyTorch wheels.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` only on your machine to configure email alerts. The `.env` file is ignored by Git.

Run the desktop webcam app:

```powershell
python main.py
```

Run the Streamlit upload app:

```powershell
python -m streamlit run streamlit_app.py
```

Press `q` in the desktop camera window to stop it. The Streamlit app processes uploaded videos; a hosted server cannot access your computer's webcam.

## Email configuration

For Gmail, use a dedicated Google App Password (with 2-Step Verification enabled), not your normal account password. Put credentials only in your local `.env` file:

```dotenv
EMAIL_FROM=your-account@gmail.com
EMAIL_TO=recipient@example.com
EMAIL_APP_PASS=your-google-app-password
ALERT_COOLDOWN_SECONDS=60
```

Email sending runs in a background thread. Alerts of the same type are limited to one every 60 seconds by default. Streamlit Cloud does not send email alerts in the current uploaded-video workflow.

## Anomaly detection status

Anomaly detection is **experimental and disabled until trained weights are supplied** at `models/autoencoder.pth`. The autoencoder must be trained on representative normal footage and evaluated on separate normal and anomalous validation footage. The current `0.015` threshold is only a placeholder; no accuracy, false-positive rate, or calibrated threshold is claimed. Do not rely on this component for safety-critical decisions.

## Streamlit Community Cloud

1. Push the repository to GitHub, including `models/yolov8n.pt`; exclude credentials and private footage.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository's `main` branch and set the main file to `streamlit_app.py`.
3. Cloud installs Python packages from `requirements.txt` and OpenCV system libraries from `packages.txt`.

The app accepts MP4, AVI, MOV, and MKV uploads up to 100 MB. Increase the frame interval or lower the video resolution for faster CPU processing. Cloud storage is ephemeral; the uploaded video is written to a temporary file and removed after analysis.

## Demo

[Open the live Streamlit app](https://intelligent-video-surveillance-dszg95tugxibpkql6bxd5b.streamlit.app/). No surveillance footage is included in this repository; only add a screenshot or GIF after confirming it contains no private or identifiable footage.

## Project files

- `main.py` — desktop webcam/RTSP entry point
- `streamlit_app.py` — uploaded-video Streamlit interface
- `camera_stream.py` — camera/RTSP capture
- `detector.py` — YOLO person detection
- `tracker.py` — DeepSORT tracking
- `motion.py` — Farneback optical-flow motion detection
- `anomaly.py` — experimental autoencoder
- `alerts.py` — environment-configured, throttled email alerts
- `database.py` — SQLite event logging
- `models/yolov8n.pt` — YOLO weights
- `logs/` — runtime SQLite event database
