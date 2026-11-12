import os
import sqlite3
from datetime import datetime

DB_PATH = "logs/events.db"


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event     TEXT,
            score     REAL,
            person_id INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_event(event, score=None, person_id=None):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (timestamp, event, score, person_id) VALUES (?,?,?,?)",
        (datetime.now().isoformat(), event, score, person_id)
    )
    conn.commit()
    conn.close()