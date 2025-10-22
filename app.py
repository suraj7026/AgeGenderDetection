import os

# Ensure Keras uses the TensorFlow backend (matches Colab setup)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import cv2
import numpy as np
import streamlit as st
import keras
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

# Set page config
st.set_page_config(
    page_title="Face Detection with Age & Gender",
    page_icon="👤",
    layout="centered",
)

st.title("Face Detection with Age & Gender 👤")
st.write("Using OpenCV Haar Cascade for face detection")
st.write("---")

# Initialize MTCNN face detector
@st.cache_resource
def load_detector():
    try:
        return MTCNN()
    except Exception as e:
        st.error(f"Error loading MTCNN detector: {e}")
        return None

detector = load_detector()

# Load age and gender models
@st.cache_resource
def load_models():
    """Load the pre-trained age and gender models."""
    try:
        age_model = load_model('agemodel.h5')
        gender_model = load_model('gendermodel.h5')
        return age_model, gender_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure 'agemodel.h5' and 'gendermodel.h5' are in the same directory as app.py.")
        return None, None

# Load models
age_model, gender_model = load_models()

if detector is None or age_model is None or gender_model is None:
    st.stop()

st.success("✅ Models loaded successfully!")


def preprocess_face(face_roi: np.ndarray, target_size: tuple[int, int]):
    """Resize face ROI and prepare batch dimension for the model."""
    if face_roi.size == 0:
        return None

    resized_face = cv2.resize(face_roi, target_size)
    return resized_face.reshape(1, *target_size, 3)


def predict_age_gender(face_roi: np.ndarray):
    """Predict age and gender for a given face ROI."""
    age_input = preprocess_face(face_roi, (200, 200))
    if age_input is None:
        return None, None

    predicted_age = age_model.predict(age_input, verbose=0)[0][0]
    predicted_age = int(np.clip(predicted_age, 0, 100))

    gender_input = preprocess_face(face_roi, (128, 128))
    if gender_input is None:
        return predicted_age, None

    gender_prob = gender_model.predict(gender_input, verbose=0)[0][0]
    predicted_gender = "Female" if gender_prob > 0.5 else "Male"
    return predicted_age, predicted_gender


def draw_label(image: np.ndarray, box: tuple[int, int, int, int], label: str, color: tuple[int, int, int]):
    """Draw bounding box and label on the image."""
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top_left = (x, y - label_size[1] - 10)
    bottom_right = (x + label_size[0], y - 10)
    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
    cv2.putText(image, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


if age_model is not None and gender_model is not None:
    st.success("✅ Models loaded successfully!")

# --- Streamlit App Logic ---

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
info_display = st.empty()
predictions_display = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning("Could not open webcam.")
    else:
        st.success("Webcam started! Looking for faces...")
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = detector.detect_faces(frame_rgb)

            predictions_lines = []

            for face in detected_faces:
                if face.get("confidence", 0) < 0.9:
                    continue

                x, y, width, height = face["box"]
                x, y = abs(x), abs(y)
                x2, y2 = x + width, y + height

                face_roi = frame_rgb[y:y2, x:x2]
                if face_roi.size == 0:
                    continue

                age, gender = predict_age_gender(face_roi)

                if age is None or gender is None:
                    label = "Face detected"
                    color = (0, 255, 0)
                else:
                    label = f"{gender}, {age}"
                    color = (255, 105, 180) if gender == "Female" else (65, 105, 225)
                    predictions_lines.append(f"**Face** at ({x}, {y}, {width}, {height}): {label}")

                draw_label(frame_rgb, (x, y, width, height), label, color)

            if not predictions_lines:
                predictions_display.markdown("**No predictions yet.**")
            else:
                predictions_display.markdown("\n".join(predictions_lines))

            info_display.markdown(
                f"""
                **Frame Info:**
                - Shape: `{frame_rgb.shape}`
                - Faces Detected: {len(predictions_lines)}
                - Mode: RGB
                """
            )

            FRAME_WINDOW.image(frame_rgb, channels="RGB")

        cap.release()
        st.write("Webcam stopped.")
else:
    st.info('Check the "Start Webcam" box to begin.')
