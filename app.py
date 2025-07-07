from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import json
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and class names
model = tf.keras.models.load_model("person_classifier_model.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

detector = MTCNN()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    filename = file.filename
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Detect face
    faces = detector.detect_faces(image_np)
    if not faces:
        return "No face detected", 400

    x, y, width, height = faces[0]['box']
    face = image_np[y:y+height, x:x+width]
    face_img = Image.fromarray(face).resize((128, 128))
    face_array = np.array(face_img) / 255.0
    face_array = np.expand_dims(face_array, axis=0)

    prediction = model.predict(face_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template("result.html", prediction=predicted_class, image_path=image_path)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
