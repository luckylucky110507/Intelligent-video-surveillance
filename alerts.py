import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
APP_PASS = os.getenv("EMAIL_APP_PASS")


def send_alert(event_type, score=None, frame_path=None):
    if not EMAIL_FROM or not EMAIL_TO or not APP_PASS:
        print(f"Alert skipped for '{event_type}': set EMAIL_FROM, EMAIL_TO, and EMAIL_APP_PASS to enable email alerts.")
        return

    msg = MIMEMultipart()
    msg["Subject"] = f"[IVS Alert] {event_type}"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    body = f"""
    INTELLIGENT VIDEO SURVEILLANCE ALERT
    =====================================
    Event    : {event_type}
    Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Score    : {score if score else 'N/A'}
    """
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, APP_PASS)
            server.send_message(msg)
        print(f"Alert sent: {event_type}")
    except (smtplib.SMTPAuthenticationError, smtplib.SMTPException, OSError, ValueError) as exc:
        print(f"Email alert failed for '{event_type}': {exc}")