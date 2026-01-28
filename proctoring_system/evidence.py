from db_connect import get_db
import cv2
import numpy as np
import os
import gridfs  
from pymongo import MongoClient


student_id = input("Enter the student ID to search for evidence: ").strip()

db = get_db()

# --- IMAGE Evidence (normal and GridFS)
evidence_col = db['evidence']
cursor = evidence_col.find({"student_id": student_id})
count = 0
has_images = False
for doc in cursor:
    has_images = True
    
    img_bytes = doc.get('image')
    if not img_bytes and 'image_fileid' in doc:
        fs = gridfs.GridFS(db)
        grid_out = fs.get(doc['image_fileid'])
        img_bytes = grid_out.read()
    if not img_bytes:
        print(f"Warning: Evidence record missing image bytes, skipping.")
        continue
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    count += 1
    event_type = doc.get('event_type', 'UnknownEvent')
    cv2.imshow(f"Evidence Image {count}: {event_type}", img)
    cv2.waitKey(1000)  
    cv2.destroyAllWindows()
if not has_images:
    print(f"No image evidence found for student {student_id}.")
else:
    print(f"Displayed {count} images.")

# --- AUDIO Evidence (normal and GridFS)
audio_col = db['voice_evidence']
audio_cursor = audio_col.find({"student_id": student_id})
audio_count = 0
export_dir = os.path.join("data", student_id, "exported_audio")
os.makedirs(export_dir, exist_ok=True)
has_audio = False
for doc in audio_cursor:
    has_audio = True
    # Try normal binary field first
    audio_bytes = doc.get('audio')
    if not audio_bytes and 'audio_fileid' in doc:
        fs = gridfs.GridFS(db)
        grid_out = fs.get(doc['audio_fileid'])
        audio_bytes = grid_out.read()
    if not audio_bytes:
        print(f"Warning: Evidence record missing audio bytes, skipping.")
        continue
    event = doc.get('event_type', 'UnknownEvent')
    filename = f"{event}_{audio_count}.wav"
    filepath = os.path.join(export_dir, filename)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    print(f"Exported suspicious voice: {filepath}")
    audio_count += 1
if not has_audio:
    print(f"No suspicious audio files found for student {student_id}.")
else:
    print(f"Exported {audio_count} suspicious audio files.")
