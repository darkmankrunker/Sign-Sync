import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import joblib

# -----------------------------
# Load trained model + labels
# -----------------------------
model = joblib.load("asl_model.joblib")
le = joblib.load("labels.pkl")

# -----------------------------
# MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1
)

# -----------------------------
# Feature extraction
# Same 71-feature method used
# during model training
# -----------------------------
def extract_features(coords):
    wrist = coords[0]
    norm = coords - wrist

    tips = norm[[4, 8, 12, 16, 20]]
    distances = np.linalg.norm(tips, axis=1)

    vec_index = norm[8] - norm[5]
    vec_middle = norm[12] - norm[9]
    vec_ring = norm[16] - norm[13]

    def safe_angle(v1, v2):
        dot = np.dot(v1, v2)
        norm_product = (
            np.linalg.norm(v1) *
            np.linalg.norm(v2)
        )

        cos = np.clip(
            dot / (norm_product + 1e-8),
            -1.0,
            1.0
        )

        return np.arccos(cos)

    angles = [
        safe_angle(vec_index, vec_middle),
        safe_angle(vec_middle, vec_ring),
        safe_angle(vec_index, vec_ring)
    ]

    return np.hstack([
        norm.flatten(),
        distances,
        angles
    ])


# -----------------------------
# Process webcam frame
# -----------------------------
def predict(frame):

    if frame is None:
        return None

    # Gradio gives RGB
    rgb = frame.copy()

    results = hands.process(rgb)

    output = frame.copy()

    if results.multi_hand_landmarks:

        landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(
            output,
            landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract coordinates
        coords = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ])

        # Extract the same 71 features
        features = extract_features(coords)
        features = features.reshape(1, -1)

        # Prediction
        probabilities = model.predict_proba(features)[0]

        confidence = probabilities.max()

        prediction_index = probabilities.argmax()

        letter = le.inverse_transform(
            [prediction_index]
        )[0]

        # Display prediction
        cv2.putText(
            output,
            f"{letter}",
            (40, 100),
            cv2.FONT_HERSHEY_DUPLEX,
            3,
            (0, 255, 0),
            6,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"Confidence: {confidence * 100:.1f}%",
            (40, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            output,
            "Show your hand",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


# -----------------------------
# Gradio interface
# -----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True
    ),
    outputs=gr.Image(type="numpy"),
    title="SignSync",
    description=(
        "Real-time ASL alphabet recognition using "
        "MediaPipe hand landmarks and an XGBoost model."
    ),
    live=True
)

demo.launch()
