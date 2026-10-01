# app.py → SignSync with REAL VOICE (just run pip install pyttsx3 first)
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import pyttsx3
import threading
import warnings
warnings.filterwarnings("ignore")

# Text-to-speech (works offline!)
tts = pyttsx3.init()
tts.setProperty('rate', 150)

model = joblib.load("asl_model.joblib")
le = joblib.load("labels.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       model_complexity=1)

def extract_features(coord):
    wrist = coord[0]
    norm = coord - wrist
    tips = norm[[4,8,12,16,20]]
    dists = np.linalg.norm(tips, axis=1)
    vi = norm[8]-norm[5]; vm = norm[12]-norm[9]; vr = norm[16]-norm[13]
    ang = lambda a,b: np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8), -1, 1))
    return np.hstack([norm.flatten(), dists, [ang(vi,vm), ang(vm,vr), ang(vi,vr)]])

history = deque(maxlen=15)
conf_history = deque(maxlen=20)

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
cv2.namedWindow("SignSync", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SignSync", 1280, 720)

current_word = ""
space_held = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    locked_letter = ""
    smooth_conf = 0.0

    if results.multi_hand_landmarks:
        lmks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lmks, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255,255,255), thickness=4, circle_radius=8),
            mp_draw.DrawingSpec(color=(50,50,255), thickness=6))

        coords = np.array([[l.x, l.y, l.z] for l in lmks.landmark])
        feats = extract_features(coords).reshape(1, -1)
        prob = model.predict_proba(feats)[0]
        conf = prob.max()
        pred = le.inverse_transform([prob.argmax()])[0]

        if conf > 0.92:
            history.append(pred)
            conf_history.append(conf)

    if history:
        locked_letter = max(set(history), key=history.count)
        smooth_conf = sum(conf_history)/len(conf_history)

    key = cv2.waitKey(1) & 0xFF

    # Hold SPACE → add letter on release
    if key == 32:
        space_held = True
    elif space_held and key != 32:
        if locked_letter and smooth_conf > 0.90:
            current_word += locked_letter
        space_held = False
        history.clear()
        conf_history.clear()

    if key == 8:  # Backspace
        current_word = current_word[:-1]
    if key == ord('c'):  # Clear all
        current_word = ""

    if key == ord('s') and current_word:
        threading.Thread(target=lambda: (tts.say(current_word), tts.runAndWait()), daemon=True).start()

    #PERFECT "SignSync" TITLE (top-right, anti-aliased, never cut off)
    title_text = "SignSync"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    thickness = 1
    color = (220, 100, 255)  # Beautiful purple

    # Get text size correctly
    (text_width, text_height), baseline = cv2.getTextSize(title_text, font, font_scale, thickness)

    # Auto position: 40px from right edge, 90px from top
    x = w - text_width - 40
    y = 90

    # Draw shadow (optional – looks premium)
    cv2.putText(frame, title_text, (x + 4, y + 4), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)

    # Draw main text (smooth & beautiful)
    cv2.putText(frame, title_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    if locked_letter:
        col = (70,255,150) if smooth_conf > 0.94 else (100,200,255)
        cv2.putText(frame, locked_letter, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 4, col, 8)

    cv2.rectangle(frame, (0, h-120), (w, h), (0,0,0), -1)
    cv2.putText(frame, current_word or "[spell a word...]", (50, h-40), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255,255,255), 5)

    cv2.putText(frame, "SPACE=Add | BACKSPACE=Del | S=Speak | C=Clear", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)

    cv2.imshow("SignSync", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()