import os
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
APP_PASS = os.getenv("EMAIL_APP_PASS")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))
_alert_lock = threading.Lock()
_last_alert_time = {}


def send_alert(event_type, score=None, frame_path=None):
    now = time.monotonic()
    with _alert_lock:
        last_sent = _last_alert_time.get(event_type, float("-inf"))
        if now - last_sent < ALERT_COOLDOWN_SECONDS:
            return False
        _last_alert_time[event_type] = now

    if not EMAIL_FROM or not EMAIL_TO or not APP_PASS:
        print(f"Alert skipped for '{event_type}': set EMAIL_FROM, EMAIL_TO, and EMAIL_APP_PASS to enable email alerts.")
        return True

    msg = MIMEMultipart()
    msg["Subject"] = f"[IVS Alert] {event_type}"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    body = f"""
    INTELLIGENT VIDEO SURVEILLANCE ALERT
    =====================================
    Event    : {event_type}
    Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Score    : {score if score is not None else 'N/A'}
    """
    msg.attach(MIMEText(body, "plain"))

    thread = threading.Thread(
        target=_send_message,
        args=(msg, event_type),
        name=f"email-alert-{event_type}",
        daemon=True,
    )
    thread.start()
    return True


def _send_message(msg, event_type):
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, APP_PASS)
            server.send_message(msg)
        print(f"Alert sent: {event_type}")
    except (smtplib.SMTPAuthenticationError, smtplib.SMTPException, OSError, ValueError) as exc:
        print(f"Email alert failed for '{event_type}': {exc}")