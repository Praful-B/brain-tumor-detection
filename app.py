## possibly make Grad-CAM

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("deMLon_model.h5")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_tumor(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    if prediction > 0.5:
        return "Tumor Detected", prediction * 100
    else:
        return "No Tumor", (1 - prediction) * 100


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            result, confidence = predict_tumor(image_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=False)