from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("deMLon_model.h5")

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# The name of the convolutional layer for Grad-CAM
# Using conv2d_3 for best localization of tumor regions
LAST_CONV_LAYER_NAME = 'conv2d_3'

print(f"[INFO] Model loaded. Using layer '{LAST_CONV_LAYER_NAME}' for Grad-CAM")


def generate_gradcam(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap for the given image
    """
    print(f"[DEBUG] Generating Grad-CAM using layer: {last_conv_layer_name}")
    
    # Create a model that outputs both the conv layer activations and final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary classification with sigmoid output
        loss = predictions[:, 0]
    
    # Get the gradients of the loss with respect to the conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute the guided gradients (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs by the gradients
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads
    
    # Create the heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    print(f"[DEBUG] Heatmap generated successfully! Shape: {heatmap.shape}, Min: {tf.reduce_min(heatmap):.4f}, Max: {tf.reduce_max(heatmap):.4f}")
    
    return heatmap.numpy()


def create_gradcam_overlay(img_path, heatmap, output_path):
    """
    Create a vibrant thermal-style overlay of the Grad-CAM heatmap on the original image
    """
    print(f"[DEBUG] Creating Grad-CAM overlay. Input: {img_path}, Output: {output_path}")
    
    # Load the original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image at {img_path}")
        return None
    
    original_img = img.copy()
    print(f"[DEBUG] Original image shape: {img.shape}")
    
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply thresholding to focus on important regions only
    # This removes weak activations (like the background)
    threshold = 0.3  # Only show activations above 30%
    heatmap_resized = np.where(heatmap_resized > threshold, heatmap_resized, 0)
    
    # Renormalize after thresholding
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / heatmap_resized.max()
    
    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap for vibrant thermal effect
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Create mask to only overlay where heatmap is significant
    mask = heatmap_resized > 0.1
    mask_3channel = np.stack([mask] * 3, axis=-1)
    
    # Blend: show original image where mask is False, show overlay where mask is True
    superimposed_img = original_img.copy()
    superimposed_img[mask_3channel] = cv2.addWeighted(
        original_img, 0.4, heatmap_colored, 0.6, 0
    )[mask_3channel]
    
    # Save the result
    success = cv2.imwrite(output_path, superimposed_img)
    if success:
        print(f"[DEBUG] ✓ Grad-CAM overlay saved successfully to {output_path}")
    else:
        print(f"[ERROR] ✗ Failed to save Grad-CAM overlay to {output_path}")
    
    return output_path if success else None


def predict_tumor(img_path, filename):
    """
    Predict tumor and generate Grad-CAM visualization
    """
    print(f"\n[DEBUG] ===== Starting prediction for {filename} =====")
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    print(f"[DEBUG] Prediction value: {prediction:.4f}")

    # Determine result
    if prediction > 0.5:
        result = "Tumor Detected"
        confidence = prediction * 100
    else:
        result = "No Tumor"
        confidence = (1 - prediction) * 100
    
    print(f"[DEBUG] Result: {result}, Confidence: {confidence:.2f}%")
    
    # Generate Grad-CAM for all predictions (adjust threshold if needed)
    gradcam_path = None
    should_generate = prediction > 0.2  # Generate if confidence > 20%
    print(f"[DEBUG] Should generate Grad-CAM: {should_generate} (threshold: 0.2)")
    
    if should_generate:
        try:
            # Generate heatmap
            heatmap = generate_gradcam(img_array, model, LAST_CONV_LAYER_NAME)
            
            # Create overlay
            gradcam_filename = f"gradcam_{filename}"
            gradcam_output_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
            
            result_path = create_gradcam_overlay(img_path, heatmap, gradcam_output_path)
            
            if result_path:
                gradcam_path = f"gradcam/{gradcam_filename}"
                print(f"[DEBUG] ✓ Grad-CAM path for template: {gradcam_path}")
            else:
                print("[ERROR] ✗ Grad-CAM overlay creation failed")
                
        except Exception as e:
            print(f"[ERROR] ✗ Exception during Grad-CAM generation: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[DEBUG] ===== Prediction complete. Grad-CAM path: {gradcam_path} =====\n")
    
    return result, confidence, gradcam_path


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None
    gradcam_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = file.filename
            image_path_full = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path_full)
            
            print(f"[DEBUG] File saved to: {image_path_full}")

            result, confidence, gradcam_path = predict_tumor(image_path_full, filename)
            
            # Convert to URL path for template
            image_path = f"uploads/{filename}"
            
            print(f"[DEBUG] Template variables - result: {result}, confidence: {confidence}, image_path: {image_path}, gradcam_path: {gradcam_path}")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path,
        gradcam_path=gradcam_path
    )


if __name__ == "__main__":
    app.run(debug=True)