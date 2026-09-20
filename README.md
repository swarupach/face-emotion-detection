# Face Emotion Detection using CNN

A real-time facial expression classification system built using Python, OpenCV, TensorFlow, and a Convolutional Neural Network (CNN).

The system detects a face through a webcam and classifies the visible facial expression into one of seven categories.

## Features

- Real-time face detection using a webcam
- Facial expression classification using CNN
- FER-2013 dataset preprocessing
- 48×48 grayscale image processing
- Pixel normalization
- Seven expression categories
- Real-time prediction confidence score
- OpenCV-based face detection and display

## Expression Categories

The trained model classifies facial expressions into:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Convolutional Neural Network (CNN)

## Dataset

This project uses the FER-2013 facial expression dataset.

The dataset contains grayscale facial images of size 48×48 pixels and seven expression categories.

The training portion contains 28,709 images.

### Dataset Labels

| Label | Expression |
|------:|------------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

> Note: The original FER-2013 dataset is not included in this repository. Please obtain the dataset separately according to its applicable terms.

## How the Project Works

The system follows these steps:

```text
Webcam
   ↓
Video Frame
   ↓
Face Detection
   ↓
Face Cropping
   ↓
Grayscale Conversion
   ↓
Resize to 48×48
   ↓
Pixel Normalization
   ↓
CNN Model
   ↓
Expression Prediction
   ↓
Confidence Score

CNN Architecture

The project uses a simple Convolutional Neural Network consisting of:

Input: 48 × 48 × 1
        ↓
Conv2D - 32 filters
        ↓
MaxPooling
        ↓
Conv2D - 64 filters
        ↓
MaxPooling
        ↓
Conv2D - 128 filters
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense - 128 neurons
        ↓
Dropout
        ↓
Dense - 7 classes
        ↓
Softmax Output

Project Structure

face-emotion-detection/
│
├── dataset/
│   └── fer2013.csv
│
├── images/
│   ├── training_images.npy
│   └── training_labels.npy
│
├── models/
│   └── emotion_model.keras
│
├── src/
│   ├── train_model.py
│   ├── emotion_detection.py
│   ├── face_detection.py
│   ├── test_camera.py
│   ├── explore_dataset.py
│   └── view_image.py
│
├── .gitignore
├── requirements.txt
└── README.md

Installation

1. Clone the repository

git clone https://github.com/swarupach/face-emotion-detection.git

2. Open the project

cd face-emotion-detection

3. Create a virtual environment

python -m venv .venv

4. Activate the virtual environment

On Windows:

.venv\Scripts\activate

5. Install dependencies

pip install -r requirements.txt

Training the Model

If you want to train the model yourself, place the FER-2013 dataset CSV file inside:

dataset/fer2013.csv

Then run:

python src/view_image.py

This preprocesses the training images and creates the required NumPy arrays.

Then train the CNN:

python src/train_model.py

The trained model will be saved as:

models/emotion_model.keras

Running Real-Time Detection

After the trained model is available, run:

python src/emotion_detection.py

The webcam will open and detect faces.

The system will display the predicted facial expression and confidence score above the detected face.

Press:

Q

to close the webcam window.

Example

The system can display predictions such as:

Happy (82.5%)

along with a rectangle around the detected face.

Preprocessing

The facial images are processed before being given to the CNN.

The preprocessing steps include:

1. Selecting the training images from FER-2013.


2. Converting pixel strings into numerical pixel values.


3. Reshaping images into 48×48 dimensions.


4. Converting images to grayscale format.


5. Normalizing pixel values between 0 and 1.


6. Reshaping the data for CNN input.



Model Training

The CNN is trained using:

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Output Classes: 7

Batch Size: 64

Epochs: 15

Validation Split: 10%


The best model based on validation accuracy is saved using a model checkpoint.

Applications

This type of facial expression classification can be used as a learning project for areas such as:

Human-computer interaction

Educational applications

User experience research

Interactive applications

Computer vision demonstrations


Limitations

Facial expressions can vary between people and situations. Lighting, camera quality, face angle, occlusion, and differences in facial appearance can affect predictions.

The model should therefore be considered a facial-expression classification system rather than a system that determines a person's actual emotional state.

Future Improvements

Possible improvements include:

Improve model accuracy through data augmentation

Use a deeper CNN architecture

Add real-time FPS display

Improve face detection

Add support for multiple faces

Add prediction history

Create a graphical user interface

Deploy the model as a web application

Evaluate the model on the FER-2013 test set


Author

Chodavarapu Jaya Sai Durga Swarupa

B.Tech Computer Science and Engineering

Disclaimer

This project classifies visible facial-expression categories from images or webcam frames. It does not determine a person's actual internal emotional state.

License

This project is intended for educational and learning purposes.

### One important thing before you push

Your current GitHub project contains `emotion_model.keras`, which is fine because your model is small. Your `.gitignore` prevents the large/local dataset and NumPy files from being uploaded.

After replacing the README, run:

```powershell
git add README.md
git commit -m "Update project README"
git push
