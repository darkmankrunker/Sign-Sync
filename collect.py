# collect.py → Fixed & super smooth landmark collector
import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

CSV_FILE = "asl_data/landmarks.csv"
if not os.path.exists("asl_data"):
    os.makedirs("asl_data")

# Create header if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{c}{i}" for i in range(21) for c in ["x","y","z"]] + ["label"]
        writer.writerow(header)
    print("Created landmarks.csv")

letter = input("\nLetter to collect (A/B/C...): ").upper()
if not letter:
    exit()

print(f"\nCollecting '{letter}' → Press SPACE to save, Q to quit\n")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

saved = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        row = []
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])
        row.append(letter)

        cv2.putText(frame, f"{letter}: {saved} saved", (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 3)

        key = cv2.waitKey(10) & 0xFF
        if key == 32:  # SPACE
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
            saved += 1
            print(f"Saved → {saved}")

        elif key == ord('q'):
            break
    else:
        cv2.putText(frame, "Show hand!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    cv2.imshow("Collector - SPACE = save", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\nDone! Collected {saved} samples for '{letter}'")