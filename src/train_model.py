import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load processed data
X = np.load("images/training_images.npy")
y = np.load("images/training_labels.npy")

print("Images:", X.shape)
print("Labels:", y.shape)

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(7, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save the best model
checkpoint = ModelCheckpoint(
    "models/emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# Train
history = model.fit(
    X,
    y,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    callbacks=[checkpoint]
)

print("Training completed!")
print("Best model saved to models/emotion_model.keras")