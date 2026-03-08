from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
import secrets

app = Flask(__name__)
app.secret_key = "plant_disease_secret"

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_PATH = "mobilenetv2_best.keras"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "images"), exist_ok=True)

IMG_SIZE = (224,224)

class_names = [
"Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
"Blueberry___healthy","Cherry_(including_sour)___healthy","Cherry_(including_sour)___Powdery_mildew",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
"Corn_(maize)___healthy","Corn_(maize)___Northern_Leaf_Blight",
"Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot","Peach___healthy",
"Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
"Potato___Early_blight","Potato___healthy","Potato___Late_blight",
"Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
"Strawberry___healthy","Strawberry___Leaf_scorch",
"Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___healthy",
"Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
"Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

model = None

def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully from", MODEL_PATH)

def predict_image(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]

    return predicted_class


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            prediction = predict_image(filepath)

            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(app.config['STATIC_FOLDER'], "images", static_filename)

            img = Image.open(filepath).convert("RGB")
            img.save(static_path)

            session['prediction'] = prediction
            session['image_path'] = f"images/{static_filename}"

            os.remove(filepath)

            return jsonify({'success': True})

        except Exception as e:

            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({'error': str(e)}), 500


@app.route('/result')
def result():

    prediction = session.get('prediction')
    image_path = session.get('image_path')

    if not prediction:
        return redirect(url_for('upload'))

    return render_template("result.html", prediction=prediction, image_path=image_path)


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)