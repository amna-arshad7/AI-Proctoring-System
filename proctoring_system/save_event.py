# save_event.py
from db_connect import get_db
import time

def save_event(user_id, event_type):
    db = get_db()
    events_col = db["events"]
    event = {
        "user_id": user_id,
        "event_type": event_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    events_col.insert_one(event)
    print(f"🧾 Event saved: {event_type}")
