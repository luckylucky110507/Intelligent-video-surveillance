import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

from anomaly_rules import detect_rule_anomaly
from detector import detect, draw_boxes
from motion import MotionDetector
from tracker import create_tracker, track_persons


st.set_page_config(
    page_title="Intelligent Video Surveillance",
    page_icon="📹",
    layout="wide",
)

st.title("Intelligent Video Surveillance")
st.caption("Analyze an uploaded video for people, crowding, and unusual motion.")

with st.sidebar:
    st.header("Analysis settings")
    frame_interval = st.slider(
        "Analyze every Nth frame",
        min_value=1,
        max_value=30,
        value=5,
        help="Higher values process faster but sample fewer frames.",
    )
    enable_tracking = st.checkbox("Enable person tracking", value=True)
    crowd_limit = st.number_input(
        "Crowd alert when people exceed",
        min_value=1,
        max_value=50,
        value=3,
    )
    motion_threshold = st.slider(
        "High-motion threshold",
        min_value=0.5,
        max_value=15.0,
        value=2.5,
        step=0.5,
        help="Higher thresholds reduce motion alerts. This is a scene-dependent heuristic.",
    )
    max_samples = st.number_input(
        "Maximum analyzed frames",
        min_value=10,
        max_value=1000,
        value=300,
        step=10,
    )

uploaded_video = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov", "mkv"],
    help="For cloud deployment, upload a video file. The server cannot access your computer's webcam.",
)

st.info(
    "Rule-based alerts are active for crowds and high motion. "
    "The learned autoencoder is a separate experimental feature."
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Analyze video", type="primary"):
        temp_path = None
        cap = None
        try:
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_video.getbuffer())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("Could not read this video. Try exporting it as MP4 (H.264) and uploading again.")
                st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = max(total_frames, 1)
            tracker = create_tracker() if enable_tracking else None
            motion_detector = MotionDetector(threshold=motion_threshold)
            progress = st.progress(0, text="Analyzing video...")
            preview = st.empty()
            rows = []
            sample_count = 0
            frame_index = 0
            peak_motion_since_sample = 0.0

            while sample_count < int(max_samples):
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1

                height, width = frame.shape[:2]
                scale = min(1.0, 1280 / max(width, height))
                if scale < 1:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                _, motion_magnitude = motion_detector.detect(frame)
                motion_score = (
                    float(motion_magnitude.mean())
                    if motion_magnitude is not None
                    else 0.0
                )
                peak_motion_since_sample = max(
                    peak_motion_since_sample, motion_score
                )
                if frame_index % frame_interval != 0:
                    continue

                motion_score = peak_motion_since_sample
                peak_motion_since_sample = 0.0
                results = detect(frame)
                detections = []
                people_count = len(results.boxes)
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, 0))

                if tracker is not None:
                    annotated = track_persons(
                        frame.copy(), detections, tracker_instance=tracker
                    )
                else:
                    annotated = draw_boxes(frame.copy(), results)

                anomaly, reasons = detect_rule_anomaly(
                    people_count,
                    motion_score,
                    crowd_limit=int(crowd_limit),
                    motion_threshold=float(motion_threshold),
                )
                reason_text = ", ".join(reasons)
                if anomaly:
                    cv2.putText(
                        annotated,
                        f"ANOMALY: {reason_text}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                timestamp = frame_index / fps if fps > 0 else 0
                rows.append(
                    {
                        "timestamp_seconds": round(timestamp, 2),
                        "people_detected": people_count,
                        "peak_motion_score": round(motion_score, 3),
                        "anomaly": anomaly,
                        "anomaly_reasons": reason_text,
                    }
                )
                sample_count += 1

                preview.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_index} | People: {people_count}",
                    use_container_width=True,
                )
                progress.progress(
                    min(frame_index / total_frames, 1.0),
                    text=f"Analyzed {sample_count} sampled frames",
                )

            if not rows:
                st.warning("No frames were analyzed. Check that the video is valid and the frame interval is suitable.")
            else:
                report = pd.DataFrame(rows)
                st.success(f"Analysis complete: {sample_count} frames analyzed.")
                first, second, third = st.columns(3)
                first.metric("Frames analyzed", sample_count)
                second.metric("Peak people detected", int(report["people_detected"].max()))
                anomaly_count = int(report["anomaly"].sum())
                third.metric("Anomaly frames", anomaly_count)

                st.subheader("Detection report")
                st.dataframe(report, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV report",
                    data=report.to_csv(index=False).encode("utf-8"),
                    file_name="surveillance_report.csv",
                    mime="text/csv",
                )
        except (cv2.error, OSError, RuntimeError, ValueError) as exc:
            st.error(f"Video analysis failed: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
else:
    st.info("Upload a video to start analysis.")
