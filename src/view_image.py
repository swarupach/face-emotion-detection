import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv("dataset/fer2013.csv")

# -------------------------------
# 1. Explore dataset
# -------------------------------

print("Dataset shape:", data.shape)

print("\nImages in each group:")
print(data["Usage"].value_counts())

# -------------------------------
# 2. Select training data
# -------------------------------

training_data = data[data["Usage"] == "Training"]

print("\nTraining images:", len(training_data))

# -------------------------------
# 3. Get training labels
# -------------------------------

training_labels = np.array(training_data["emotion"])

print("Label shape:", training_labels.shape)

print("First 10 labels:")
print(training_labels[:10])

# -------------------------------
# 4. Get training pixels
# -------------------------------

training_pixels = training_data["pixels"]

print("\nNumber of pixel rows:", len(training_pixels))

# -------------------------------
# 5. Convert ALL pixel strings
# -------------------------------

images = []

for pixel_string in training_pixels:

    pixels = [int(value) for value in pixel_string.split()]

    image = np.array(pixels).reshape(48, 48)

    image = image / 255.0

    images.append(image)

# Convert list to NumPy array
images = np.array(images)
images = images.reshape(-1,48,48,1)

print("\nImages array shape:", images.shape)
print("Minimum pixel:", images.min())
print("Maximum pixel:", images.max())
print("Final image shape:",images.shape)
# -------------------------------
# 6. Display first training image
# -------------------------------

plt.imshow(images[0], cmap="gray")

plt.title("Emotion: " + str(training_labels[0]))

plt.axis("off")

plt.show()

np.save("images/training_images.npy", images)
np.save("images/training_labels.npy", training_labels)

print("Training data saved successfully!")