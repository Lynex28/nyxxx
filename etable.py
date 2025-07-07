import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from numpy import asarray
from PIL import Image
from mtcnn.mtcnn import MTCNN
import json
from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle

# Set parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

# ✅ Set to your local dataset folder
DATA_DIR = r"C:\Users\User\Downloads\lynex\image_classification_dataset"

def draw_image_with_boxes(filename, result_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)

def extract_face_from_image(image_path, required_size=(128, 128)):
    image = pyplot.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_images = []

    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images

def load_data(data_dir):
    image_data = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Sorted for consistent order

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            pixels = pyplot.imread(img_path)
            detector = MTCNN()
            faces = detector.detect_faces(pixels)

            if not faces:
                continue  # Skip if no face detected

            extracted_face = extract_face_from_image(img_path)
            if not extracted_face:
                continue

            img_array = tf.keras.utils.img_to_array(cv2.resize(extracted_face[0], IMAGE_SIZE))
            image_data.append(img_array)
            labels.append(idx)

    image_data = tf.convert_to_tensor(image_data) / 255.0
    labels = tf.convert_to_tensor(labels)
    return image_data, labels, class_names

print("Loading data...")
image_data, labels, class_names = load_data(DATA_DIR)
print("Classes:", class_names)

# Save class names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Split data
X_train, X_val, y_train, y_val = train_test_split(image_data.numpy(), labels.numpy(), test_size=0.2, random_state=42)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])
    return model

model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training the model...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save model
model.save("person_classifier_model.h5")
print("Model saved as person_classifier_model.h5")
