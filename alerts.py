import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

EMAIL_FROM = os.getenv("EMAIL_FROM", "kumarilucky01437@gmail.com")
EMAIL_TO = os.getenv("EMAIL_TO", "recipient@gmail.com")
APP_PASS = os.getenv("EMAIL_APP_PASS", "LUC_ky1114")  # Prefer setting this in environment

def send_alert(event_type, score=None, frame_path=None):
    msg = MIMEMultipart()
    msg["Subject"] = f"[IVS Alert] {event_type}"
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO

    body = f"""
    INTELLIGENT VIDEO SURVEILLANCE ALERT
    =====================================
    Event    : {event_type}
    Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Score    : {score if score else 'N/A'}
    """
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, APP_PASS)
        server.send_message(msg)
    print(f"Alert sent: {event_type}")