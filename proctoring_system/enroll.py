import os
import cv2
import numpy as np
from datetime import datetime
from deepface import DeepFace
from db_connect import get_db
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
from scipy.io.wavfile import write

# --- Face Enrollment

def capture_student_info():
    student_id = input("Enter Student ID: ").strip()
    student_name = input("Enter Student Name: ").strip()
    if not student_id or not student_name:
        print("❌ Student ID and Name cannot be empty!")
        return None, None
    return student_id, student_name

def capture_images(student_folder):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Enroll Student", cv2.WINDOW_NORMAL)
    ref_images, count = [], 0
    while count < 5:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Enroll Student", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            ref_images.append(frame.copy())
            file_path = os.path.join(student_folder, f"reference_image_{count+1}.jpg")
            cv2.imwrite(file_path, frame)
            count += 1
            print(f"✅ Image {count}/5 captured!")
        elif key & 0xFF == ord('q'):
            print("❌ Enrollment cancelled.")
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(ref_images) != 5:
        print("❌ Not enough images captured!")
        return None
    return ref_images

def extract_save_embeddings(images, student_folder):
    embeddings = []
    for idx in range(len(images)):
        file_path = os.path.join(student_folder, f"reference_image_{idx+1}.jpg")
        try:
            result = DeepFace.represent(img_path=file_path, model_name="Facenet")
            if isinstance(result, list):
                embedding = result[0]["embedding"]
            else:
                embedding = result["embedding"]
            embeddings.append(np.array(embedding))
        except Exception as e:
            print(f"❌ Face not detected in image {idx+1}: {e}")
            return None
    np.save(os.path.join(student_folder, "reference_embeddings.npy"), np.array(embeddings))
    return True

# --- Voice Enrollment

def capture_voice(student_folder):
    #print("Please read aloud: 'I am ready to start my exam.'")
    fs = 16000
    seconds = 6
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio_path = os.path.join(student_folder, "reference_voice.wav")
    write(audio_path, fs, recording)
    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    return audio_path, embed

# --- Save to DB (Face & Voice)

def save_to_db(student_id, student_name, student_folder, voice_path, voice_embed):
    try:
        db = get_db()
        students_col = db["students"]
        student_doc = {
            "id": student_id,
            "name": student_name,
            "enrolled_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_folder": student_folder,
            "embeddings_path": os.path.join(student_folder, "reference_embeddings.npy"),
            "voice_embed": voice_embed.tolist(),
            "voice_path": voice_path,
            "status": "enrolled"
        }
        students_col.update_one({"id": student_id}, {"$set": student_doc}, upsert=True)
        print("📚 Student info (face/voice) saved in MongoDB!")
    except Exception as e:
        print("❌ MongoDB error:", e)

# --- Main Enrollment Process

def main():
    student_id, student_name = capture_student_info()
    if not student_id or not student_name:
        return
    student_folder = os.path.join("data", student_id)
    os.makedirs(student_folder, exist_ok=True)
    images = capture_images(student_folder)
    if not images:
        return
    if not extract_save_embeddings(images, student_folder):
        return
    voice_path, voice_embed = capture_voice(student_folder)
    save_to_db(student_id, student_name, student_folder, voice_path, voice_embed)
    print(f"✅ Enrollment complete for {student_name} ({student_id})")

if __name__ == "__main__":
    main()
