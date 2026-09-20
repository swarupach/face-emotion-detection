# Face Emotion Detection using CNN

A real-time facial expression classification project using Python, OpenCV, TensorFlow, and the FER-2013 dataset.

## Features

- Real-time webcam face detection
- Facial expression classification
- CNN-based deep learning model
- Grayscale 48x48 image preprocessing
- Pixel normalization
- Confidence score display

## Expressions

The model classifies facial expressions into:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Technologies

- Python
- OpenCV
- NumPy
- Pandas
- TensorFlow / Keras
- Matplotlib
- CNN

## Project Structure

```text
emotion_recognition/
├── dataset/
│   └── fer2013.csv
├── images/
│   ├── training_images.npy
│   └── training_labels.npy
├── models/
│   └── emotion_model.keras
├── src/
│   ├── train_model.py
│   ├── emotion_detection.py
│   ├── face_detection.py
│   └── test_camera.py
├── requirements.txt
└── README.md

