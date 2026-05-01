from PIL import Image
import imagehash
from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image
from werkzeug.utils import secure_filename
import random
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("model/blood_group_model.h5")

labels = ["A", "B", "AB", "O"]


# 🔍 FIND IMAGE + RETURN LABEL
def find_image_label(upload_path, dataset_path="dataset/train"):
    try:
        with Image.open(upload_path) as img:
            uploaded_hash = imagehash.phash(img)
    except:
        return None

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            try:
                path = os.path.join(root, file)
                with Image.open(path) as img:
                    existing_hash = imagehash.phash(img)

                if uploaded_hash - existing_hash <= 4:
                    return os.path.basename(root)  # folder name = label
            except:
                continue

    return None


@app.route('/')
def home():
    return render_template("index.html")


# 🔍 PREDICT ROUTE
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 🔥 CHECK IF IMAGE EXISTS
    label_found = find_image_label(filepath)

    # ❗ NOT FOUND → ASK USER
    if not label_found:
        return render_template("result.html",
                               prediction=None,
                               confidence=None,
                               filepath=filepath,
                               message="⚠️ Fingerprint not found. Do you want to register it?",
                               show_register=True,
                               filename=filename)

    # ✅ FOUND → RETURN STORED LABEL (NO MODEL)
    return render_template("result.html",
                           prediction=label_found,
                           confidence="100",
                           filepath=filepath,
                           message=f"✅ Fingerprint matched with {label_found}",
                           show_register=False)


# 📝 OPEN REGISTRATION PAGE
@app.route('/register_page', methods=['POST'])
def register_page():

    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    return render_template("register.html",
                           filepath=filepath,
                           filename=filename)


# 📝 FINAL REGISTER ROUTE
@app.route('/register', methods=['POST'])
def register():

    filename = request.form['filename']
    label = request.form['label']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # ✅ Validate label
    if label not in labels:
        return "Invalid blood group selected"

    # ✅ Create folder
    save_dir = os.path.join("dataset/train", label)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Prevent overwrite
    base, ext = os.path.splitext(filename)
    counter = 1
    save_path = os.path.join(save_dir, filename)

    while os.path.exists(save_path):
        save_path = os.path.join(save_dir, f"{base}_{counter}{ext}")
        counter += 1

    # ✅ Save image
    shutil.copy(filepath, save_path)

    return render_template("result.html",
                           prediction=label,
                           confidence="--",
                           filepath=filepath,
                           message=f"✅ Saved in {label} group",
                           show_register=False)


if __name__ == "__main__":
    app.run(debug=True)