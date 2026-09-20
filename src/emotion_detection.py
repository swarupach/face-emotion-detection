import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/emotion_model.keras")

# Emotion labels
emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Camera
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for x, y, width, height in faces:

        # Crop face
        face = gray[y:y + height, x:x + width]

        # Resize
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face / 255.0

        # Reshape for CNN
        face = face.reshape(1, 48, 48, 1)

        # Predict
        predictions = model.predict(face, verbose=0)

        emotion_index = np.argmax(predictions)
        emotion = emotions[emotion_index]

        confidence = predictions[0][emotion_index] * 100

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Display prediction
        text = f"{emotion} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Expression Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()