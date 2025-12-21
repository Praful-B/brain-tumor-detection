# INSTALLING THE DEPENDANCIES

from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.models import Model 
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok = True)
os.makedirs(RESULT_FOLDER, exist_ok = True)

# LOADING THE MODEL

model = tf.keras.models.load_model("brain_tumor_sl_model.keras")

# GRAD-CAM FUNCTION

def makeGradCAM_heatmap(imgArray, model) :
    gradModel = Model(model.inputs,[model.get_layer("last_conv").output,model.output[0]])
    with tf.GradientTape() as tape :
        conv_out, preds = gradModel(imgArray)
        loss = preds[:,0]
    
    grads = tape.gradient(loss,conv_out)
    pooled_grads = tf.reduce_mean(grads,axis = (0,1,2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis = 1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# ROUTES

@app.route("/",methods = ["GET","POST"])
def index() :
    if request.method == "POST" :
        file = request.files["image"]
        if file :
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Reading the image

            img = cv2.imgread(filepath)
            img = cv2.resize(img,(256,256))
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_normal = imgRGB / 255.0

            imgArray = np.expand_dims(img_normal, axis = 0)

            # Prediction 

            tumorPred, severityPred = model.predict(imgArray)

            tumorProb = float(tumorPred[0][0])
            tumorDetected = tumorProb >= 0.35
            severityIDX = int(np.argmax(severityPred[0]))
            severityConfi = float(np.max(severityPred[0]))

            severityMAp = {0 : "MILD",1 : "HARMFUL",2 : "DANGEROUS"}

            # GRAD-CAM

            heatmap = makeGradCAM_heatmap(imgArray, model)
            heatmap = cv2.resize(heatmap,(256, 256))

            heatmap_colored = cv2.applyColorMap(np.uint(255 * heatmap),cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = np.uint8(0.6*imgRGB + 0.4*heatmap_colored)

            result_path = os.path.join(RESULT_FOLDER, "result.png")
            plt.imsave(result_path, overlay)

            return render_template("index.html",
                                   uploaded_image = filepath,
                                   result_image = result_path,
                                   tumor_detected = tumorDetected,
                                   tumor_conf = round(tumorProb * 100, 2),
                                   severity = severityMAp[severityIDX],
                                   severity_conf = round(severityConfi*100, 2))
        return render_template("index.html")
    
if __name__ == "main" : 
    app.run(debug = True)
