# ============================================================
# BRAIN TUMOR DETECTION & SEVERITY PREDICTION (SL MODEL)
# ============================================================

import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt
import shutil

# ============================================================
# GPU CONFIG (Prevent OOM)
# ============================================================

gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# ============================================================
# BASIC INFO
# ============================================================

print("~" * 60)
print("BRAIN TUMOR DETECTION & SEVERITY PREDICTION")
print("SUPERVISED LEARNING CNN MODEL")
print("~" * 60)

# ============================================================
# DATASET CONFIG (UNCHANGED)
# ============================================================

severity_mapping = {
    'pituitary': 0,
    'meningioma': 1,
    'glioma': 2,
    'noTumor': -1
}

severity_labels = {
    0: "MILD",
    1: "HARMFUL",
    2: "DANGEROUS",
    -1: "NO TUMOR"
}

src_data = "C:/Users/bellu/Downloads/CrumboPersono/archive (4)"

if not os.path.exists(src_data):
    raise FileNotFoundError("Dataset path not found!")

print("✓ Dataset located")

# ============================================================
# DATASET REORGANIZATION (SEVERITY BASED)
# ============================================================

def reorganize_dataset():
    output_base = "Training_Severity"

    if os.path.exists(output_base):
        print("✓ Reorganized dataset already exists")
        return output_base

    print("Reorganizing dataset by severity levels...")

    for split in ["Training", "Testing"]:
        split_path = os.path.join(src_data, split)
        if not os.path.exists(split_path):
            continue

        for tumor_type, sev_level in severity_mapping.items():
            src_folder = os.path.join(split_path, tumor_type)
            if not os.path.exists(src_folder):
                continue

            sev_name = severity_labels[sev_level]
            dst_folder = os.path.join(output_base, split, sev_name)
            os.makedirs(dst_folder, exist_ok=True)

            for img in os.listdir(src_folder):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(
                        os.path.join(src_folder, img),
                        os.path.join(dst_folder, img)
                    )

    print("✓ Dataset reorganization complete")
    return output_base


dataDir = reorganize_dataset()

# ============================================================
# IMAGE VALIDATION & CLEANING
# ============================================================

print("Verifying images...")

imgExts = ['jpeg', 'jpg', 'png', 'bmp']
deleted = 0

for split in ["Training", "Testing"]:
    split_path = os.path.join(dataDir, split)
    if not os.path.exists(split_path):
        continue

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue

        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            try:
                if imghdr.what(img_path) not in imgExts:
                    os.remove(img_path)
                    deleted += 1
                    continue

                if cv2.imread(img_path) is None:
                    os.remove(img_path)
                    deleted += 1
            except:
                os.remove(img_path)
                deleted += 1

print(f"✓ Corrupted images removed: {deleted}")

# ============================================================
# DATA LOADING (SUPERVISED)
# ============================================================

training_path = os.path.join(dataDir, "Training")

dataset = tf.keras.utils.image_dataset_from_directory(
    training_path,
    image_size=(256, 256),
    batch_size=32,
    label_mode="int",
    shuffle=True,
    seed=42
)

class_names = dataset.class_names
print("Classes detected:", class_names)

dataset = dataset.map(lambda x, y: (x / 255.0, y))
dataset = dataset.cache().shuffle(1000)

# ============================================================
# DATA SPLIT (TRAIN / VAL / TEST)
# ============================================================

data_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.7 * data_size)
val_size = int(0.2 * data_size)
test_size = data_size - train_size - val_size

train = dataset.take(train_size)
val = dataset.skip(train_size).take(val_size)
test = dataset.skip(train_size + val_size)

train = train.prefetch(tf.data.AUTOTUNE)
val = val.prefetch(tf.data.AUTOTUNE)
test = test.prefetch(tf.data.AUTOTUNE)

# ============================================================
# LABEL PREPARATION (MULTI-TASK SL)
# ============================================================

def prepare_labels(images, labels):
    # labels:
    # 0 = DANGEROUS
    # 1 = HARMFUL
    # 2 = MILD
    # 3 = NO TUMOR

    tumor_presence = tf.cast(labels != 3, tf.float32)

    severity = tf.where(
        labels == 3,
        tf.zeros_like(labels),
        tf.where(labels == 2, 0,
        tf.where(labels == 1, 1, 2))
    )

    return images, {
        "tumor_presence": tumor_presence,
        "severity": severity
    }

train = train.map(prepare_labels)
val = val.map(prepare_labels)
test = test.map(prepare_labels)

# ============================================================
# CNN MODEL (SUPERVISED LEARNING)
# ============================================================

from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization, Input
)
from tensorflow.keras.models import Model

inputs = Input(shape=(256, 256, 3))

x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(128, 3, activation='relu', padding='same', name="last_conv")(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

tumor_output = Dense(1, activation='sigmoid', name="tumor_presence")(x)
severity_output = Dense(3, activation='softmax', name="severity")(x)

model = Model(inputs, [tumor_output, severity_output])

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "tumor_presence": "binary_crossentropy",
        "severity": "sparse_categorical_crossentropy"
    },
    metrics={
        "tumor_presence": "accuracy",
        "severity": "accuracy"
    }
)

model.summary()

# ============================================================
# TRAINING
# ============================================================

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5)
]

history = model.fit(
    train,
    validation_data=val,
    epochs=30,
    callbacks=callbacks
)

# ============================================================
# EVALUATION
# ============================================================

results = model.evaluate(test, verbose=0)

print("\nTEST RESULTS")
print(f"Total Loss:               {results[0]*100:.4f}%")
print(f"Tumor Detection Accuracy: {results[3]*100:.2f}%")
print(f"Severity Accuracy:        {results[4]*100:.2f}%")

# ============================================================
# SAVE MODEL
# ============================================================

model.save("brain_tumor_sl_model.keras")
print("✓ Model saved successfully")

# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(img_array, model):
    grad_model = Model(
        model.inputs,
        [model.get_layer("last_conv").output, model.output[0]]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_brain_tumor(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    img_array = np.expand_dims(img, axis=0)
    tumor_pred, severity_pred = model.predict(img_array, verbose=0)

    tumor_prob = float(tumor_pred[0][0])
    tumor_detected = tumor_prob >= 0.5

    sev_idx = int(np.argmax(severity_pred[0]))
    sev_conf = float(np.max(severity_pred[0]))

    sev_map = {0: "MILD", 1: "HARMFUL", 2: "DANGEROUS"}

    heatmap = make_gradcam_heatmap(img_array, model)

    return tumor_detected, tumor_prob, sev_map[sev_idx], sev_conf, heatmap

# ============================================================
# USER INPUT & VISUAL PREDICTION OUTPUT
# ============================================================

print("\n" + "="*60)
print("BRAIN TUMOR PREDICTION (USER INPUT)")
print("="*60)

def visualize_prediction(img_path, model):
    if not os.path.exists(img_path):
        print("❌ Invalid image path")
        return

    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0

    img_array = np.expand_dims(img_norm, axis=0)

    # Model prediction
    tumor_pred, severity_pred = model.predict(img_array, verbose=0)

    tumor_prob = float(tumor_pred[0][0])
    tumor_detected = tumor_prob >= 0.35

    severity_idx = int(np.argmax(severity_pred[0]))
    severity_conf = float(np.max(severity_pred[0]))

    severity_map = {
        0: "MILD",
        1: "HARMFUL",
        2: "DANGEROUS"
    }

    severity_label = severity_map[severity_idx]

    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    alpha = 0.4
    overlay = np.uint8(img_rgb * (1 - alpha) + heatmap_colored * alpha)

    # Display result
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")

    if tumor_detected:
        title_text = (
            f"TUMOR DETECTED\n"
            f"Confidence: {tumor_prob*100:.2f}%\n"
            f"Severity: {severity_label} ({severity_conf*100:.2f}%)"
        )
    else:
        title_text = (
            f"NO TUMOR DETECTED\n"
            f"Confidence: {(1 - tumor_prob)*100:.2f}%"
        )

    plt.title(title_text, fontsize=12, fontweight="bold")
    plt.show()

    # Console output (for judges)
    print("\nPrediction Summary:")
    print("Tumor Detected :", tumor_detected)
    print(f"Tumor Confidence : {tumor_prob*100:.2f}%")
    if tumor_detected:
        print(f"Severity : {severity_label}")
        print(f"Severity Confidence : {severity_conf*100:.2f}%")

# ============================================================
# TAKE USER INPUT
# ============================================================

while True:
    test_img_path = input("\nEnter MRI image path (or type 'exit'): ")

    if test_img_path.lower() == "exit":
        print("Exiting prediction mode.")
        break

    visualize_prediction(test_img_path, model)
