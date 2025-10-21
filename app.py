import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Real-time Age & Gender Detection",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="auto",
)

# Try to import MTCNN with error handling
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError as e:
    st.error(f"MTCNN import failed: {e}")
    MTCNN_AVAILABLE = False

@st.cache_resource
def load_models():
    """Load all the required models and return them."""
    try:
        age_model = load_model("agemodel.h5", compile=False)
        gender_model = load_model("gendermodel.h5", compile=False)
        
        if MTCNN_AVAILABLE:
            detector = MTCNN()
        else:
            detector = None
            
        return detector, age_model, gender_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error(
            "Please ensure 'agemodel.h5' and 'gendermodel.h5' are in the same directory as app.py."
        )
        return None, None, None


st.title("Real-time Age & Gender Detection 👤")
st.write(
    "This application uses your webcam to detect faces and predict their age and gender in real-time."
)
st.write("---")

# Check if MTCNN is available
if not MTCNN_AVAILABLE:
    st.error("MTCNN is not available. Please install it with: pip install mtcnn")
    st.stop()

# Load the models
detector, age_model, gender_model = load_models()

if detector is None:
    st.stop()

# --- Helper Functions from the Notebook ---


def preprocess_face(face_roi, target_size):
    """Resizes and reshapes a face region for model prediction."""
    if face_roi.size == 0:
        return None
    # Ensure the ROI is a PIL Image
    if isinstance(face_roi, np.ndarray):
        face_roi = Image.fromarray(face_roi)

    resized_face = face_roi.resize(target_size)
    face_array = np.asarray(resized_face)

    # Reshape for model input (add batch dimension)
    return face_array.reshape(1, *target_size, 3)


def predict_age_gender(face_roi, age_model, gender_model):
    """Predicts age and gender from a face region."""
    # Preprocess for age model
    age_input = preprocess_face(face_roi, (200, 200))
    if age_input is None:
        return "N/A", "N/A"

    # The age prediction from the notebook seems to be flawed (e.g., predicting 4000+).
    # We will clip the prediction to a more reasonable range (0-100).
    predicted_age = int(age_model.predict(age_input, verbose=0)[0][0])
    predicted_age = np.clip(predicted_age, 0, 100)

    # Preprocess for gender model
    gender_input = preprocess_face(face_roi, (128, 128))
    if gender_input is None:
        return predicted_age, "N/A"

    gender_prob = gender_model.predict(gender_input, verbose=0)[0][0]
    predicted_gender = "Female" if gender_prob > 0.5 else "Male"

    return predicted_age, predicted_gender


def draw_results(image, faces_data):
    """Draws bounding boxes and labels on the image."""
    for face_info in faces_data:
        x, y, width, height = face_info["box"]
        age, gender = face_info["age"], face_info["gender"]

        label = f"{gender}, {age}"

        # Define color based on gender
        color = (
            (255, 182, 193) if gender == "Female" else (135, 206, 235)
        )  # Pink for Female, Blue for Male

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            image,
            (x, y - label_size[1] - 10),
            (x + label_size[0], y - 10),
            color,
            cv2.FILLED,
        )

        # Draw label text
        cv2.putText(
            image, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    return image


# --- Streamlit App Logic ---

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning(
            "Could not open webcam. Please make sure it's connected and not in use by another application."
        )
    else:
        st.success("Webcam started successfully! Looking for faces...")
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam. Please restart the app.")
                break

            # Convert frame from BGR (OpenCV default) to RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            detected_faces = detector.detect_faces(frame_rgb)

            faces_data = []
            # Process each face
            for face in detected_faces:
                if face["confidence"] > 0.9:  # Process only confident detections
                    x, y, width, height = face["box"]
                    # Ensure coordinates are positive
                    x1, y1 = abs(x), abs(y)
                    x2, y2 = x1 + width, y1 + height

                    # Extract face ROI
                    face_roi = frame_rgb[y1:y2, x1:x2]

                    if face_roi.size > 0:
                        age, gender = predict_age_gender(
                            face_roi, age_model, gender_model
                        )
                        face_data = {
                            "box": (x, y, width, height),
                            "age": age,
                            "gender": gender,
                        }
                        faces_data.append(face_data)

            # Draw results on the original BGR frame
            result_image = draw_results(frame, faces_data)

            # Display the final image in the Streamlit app
            FRAME_WINDOW.image(result_image, channels="BGR")

        cap.release()
        st.write("Webcam stopped.")
else:
    st.info('Check the "Start Webcam" box to begin real-time detection.')
