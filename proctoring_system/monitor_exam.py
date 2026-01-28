import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from ultralytics import YOLO
import time
import os
import threading
import speech_recognition as sr
import datetime
from save_event import save_event
from resemblyzer import VoiceEncoder, preprocess_wav
import tempfile
from db_connect import get_db
import pygetwindow as gw
import pyautogui

# --- Evidence image saving ---
def save_image_evidence(student_id, event_type, frame):
    db = get_db()
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = buffer.tobytes()
    evidence_col = db['evidence']
    evidence_doc = {
        "student_id": student_id,
        "event_type": event_type,
        "timestamp": datetime.datetime.now(),
        "image": img_bytes
    }
    evidence_col.insert_one(evidence_doc)
    print(f"🧾 Evidence image saved for {event_type}")

# --- Evidence audio saving ---
def save_audio_evidence(student_id, event_type, audio_bytes):
    evidence_audio_folder = os.path.join("data", student_id, "suspicious_audio")
    os.makedirs(evidence_audio_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = os.path.join(evidence_audio_folder, f"{event_type}_{timestamp}.wav")
    with open(audio_file_path, "wb") as f:
        f.write(audio_bytes)
    print(f"🔊 Suspicious audio saved: {audio_file_path}")
    db = get_db()
    audio_col = db["voice_evidence"]
    audio_doc = {
        "student_id": student_id,
        "event_type": event_type,
        "timestamp": datetime.datetime.now(),
        "audio": audio_bytes
    }
    audio_col.insert_one(audio_doc)

# --- Face embeddings loader ---
def load_reference_embeddings(student_id):
    path = f"data/{student_id}/reference_embeddings.npy"
    if not os.path.exists(path):
        print("❌ No reference embeddings found! Run enrollment first.")
        return None
    return np.load(path)

# --- Cosine similarity for verification ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_embeddings(ref_embeddings, test_embedding):
    sims = [cosine_similarity(ref, test_embedding) for ref in ref_embeddings]
    best_sim = max(sims)
    return best_sim

# --- Event logging with cooldown ---
def log_event_once(user_id, event_type, last_event_time, cooldown=8):
    now = time.time()
    if event_type not in last_event_time or (now - last_event_time[event_type]) > cooldown:
        save_event(user_id, event_type)
        last_event_time[event_type] = now
        return True
    return False

# --- Gaze direction estimate ---
def estimate_gaze(face):
    x, y, w, h = face['box']
    left_eye = face['keypoints']['left_eye']
    right_eye = face['keypoints']['right_eye']
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    face_center_x = x + w / 2
    if eye_center_x < face_center_x - w * 0.1:
        return "Looking Left"
    elif eye_center_x > face_center_x + w * 0.1:
        return "Looking Right"
    else:
        return "Looking Forward"

# --- Speaker verification + audio logging ---
def audio_monitor_thread(student_id, last_event_time, enrolled_voice, stop_event, cooldown=8):
    encoder = VoiceEncoder()
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🔊 Audio monitoring started...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    while not stop_event.is_set():
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.write(audio.get_wav_data())
            temp_wav.close()
            wav = preprocess_wav(temp_wav.name)
            live_embed = encoder.embed_utterance(wav)
            if enrolled_voice is not None:
                similarity = np.dot(live_embed, enrolled_voice)/(np.linalg.norm(live_embed)*np.linalg.norm(enrolled_voice))
                print(f"Speaker similarity: {similarity:.3f}")
                if similarity < 0.75:
                    print("🛑 Suspicious non-student voice!")
                    if log_event_once(student_id, "Suspicious: Unmatched Voice", last_event_time, cooldown):
                        save_audio_evidence(student_id, "SuspiciousVoice", audio.get_wav_data())
            os.remove(temp_wav.name)
        except Exception as e:
            print("Error in audio capture or speaker verification:", e)

# --- Anti-app switching detection with screenshot evidence ---
def app_switch_monitor_thread(student_id, last_event_time, stop_event, cooldown=8):
    EXAM_WINDOW_TITLE = "Exam Monitoring"  
    while not stop_event.is_set():
        try:
            active = gw.getActiveWindow()
            title = active.title if active else "Unknown"
            if title != EXAM_WINDOW_TITLE:
                print(f"App switch detected! Active window: {title}")
                if log_event_once(student_id, f"Suspicious: App Switch ({title})", last_event_time, cooldown):
                    screenshot = pyautogui.screenshot()
                    screenshot_np = np.array(screenshot)
                    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
                    save_image_evidence(student_id, f"Suspicious: App Switch ({title})", screenshot_np)
            time.sleep(1)
        except Exception as e:
            print("App switch detection error:", e)

def main():
    student_id = input("Enter Student ID: ").strip()
    db = get_db()
    student = db["students"].find_one({"id": student_id})
    enrolled_voice = np.array(student.get("voice_embed")) if student and "voice_embed" in student else None
    ref_embeddings = load_reference_embeddings(student_id)
    if ref_embeddings is None:
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return
    cv2.namedWindow("Exam Monitoring", cv2.WINDOW_NORMAL)
    detector = MTCNN()
    yolo_model = YOLO('yolov8n.pt')
    last_event_time = {}
    print("🎥 Monitoring started... Press 'Q' to quit exam monitoring.")

    stop_event = threading.Event()
    audio_thread = threading.Thread(target=audio_monitor_thread, args=(student_id, last_event_time, enrolled_voice, stop_event))
    audio_thread.daemon = True
    audio_thread.start()
    app_thread = threading.Thread(target=app_switch_monitor_thread, args=(student_id, last_event_time, stop_event))
    app_thread.daemon = True
    app_thread.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame not received from camera.")
                break
            # Device detection with confidence threshold
            yolo_results = yolo_model(frame, conf=0.6)[0] 
            for obj in yolo_results.boxes:
                if obj.conf < 0.6:
                    continue
                label = yolo_results.names[int(obj.cls)]
                if label in ['cell phone', 'phone', 'headset', 'earphone', 'laptop', 'tablet', 'smartwatch']:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, f'Device: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if log_event_once(student_id, f'Suspicious: Device Detected ({label})', last_event_time):
                        save_image_evidence(student_id, f'Suspicious: Device Detected ({label})', frame)
            # Multi-face + gaze + verification
            faces = detector.detect_faces(frame)
            if len(faces) == 0:
                label = "❌ No Face Detected"
                if log_event_once(student_id, "No Face Detected", last_event_time):
                    save_image_evidence(student_id, "No Face Detected", frame)
                cv2.putText(frame, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if len(faces) > 1:
                    label = "🚨 Multiple Faces Detected!"
                    if log_event_once(student_id, "Suspicious: Multiple Faces", last_event_time):
                        save_image_evidence(student_id, "Suspicious: Multiple Faces", frame)
                for idx, face in enumerate(faces):
                    x, y, w, h = face['box']
                    face_img = frame[y:y+h, x:x+w]
                    try:
                        result = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
                        if isinstance(result, list):
                            test_embedding = result[0]['embedding']
                        else:
                            test_embedding = result['embedding']
                        sim = compare_embeddings(ref_embeddings, test_embedding)
                        if sim > 0.7:
                            label = f"✅ Verified ({sim:.2f})"
                        else:
                            label = f"⚠️ Unknown Face {idx+1}!"
                            if log_event_once(student_id, f"Suspicious: Unknown Face {idx+1}", last_event_time):
                                save_image_evidence(student_id, f"Suspicious: Unknown Face {idx+1}", frame)
                        gaze = estimate_gaze(face)
                        if gaze != "Looking Forward":
                            if log_event_once(student_id, f"Suspicious: {gaze}", last_event_time):
                                save_image_evidence(student_id, f"Suspicious: {gaze}", frame)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if "Verified" in label else (0, 0, 255), 2)
                        cv2.putText(frame, f"{label}, {gaze}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    except Exception as e:
                        label = f"❌ No Face Detected ({idx+1})"
                        if log_event_once(student_id, f"No Face Detected ({idx+1})", last_event_time):
                            save_image_evidence(student_id, f"No Face Detected ({idx+1})", frame)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Exam Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Monitoring stopped by user.")
                break
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
