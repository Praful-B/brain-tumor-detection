# 🧠 Brain Tumor Detection System

A Deep Learning application capable of detecting brain tumors from MRI scans. This project utilizes a Convolutional Neural Network (CNN) for classification and provides a user-friendly web interface powered by Flask.

## Overview

Early detection of brain tumors is critical for effective treatment. This project automates the detection process using Computer Vision and Deep Learning. Users can upload an MRI image via a web interface, and the system predicts whether a tumor is present along with a confidence score.

## Features

- **End-to-End Pipeline:** From data preprocessing to model deployment.
- **Custom CNN Architecture:** Built from scratch using TensorFlow/Keras.
- **Data Cleaning:** Automatic removal of corrupted images during training.
- **Web Interface:** Clean, responsive UI for easy interaction.
- **Real-time Prediction:** Instant analysis of uploaded MRI scans.
- **Visualizations:** Training accuracy/loss graphs and prediction confidence.

## Model Architecture

The model is a Sequential CNN designed for binary classification:

1.  **Input Layer:** Accepts images resized to 256x256 pixels (RGB).
2.  **Convolutional Blocks:**
    - 3 Blocks of Conv2D layers (Filters: 16, 32, 16) with ReLU activation.
    - MaxPooling2D for downsampling.
3.  **Fully Connected Layers:**
    - Flatten layer.
    - Dense layer (256 units, ReLU).
    - Dropout (0.5) to prevent overfitting.
4.  **Output Layer:** Dense layer (1 unit, Sigmoid) for binary probability (Tumor vs. No Tumor).

_Optimizer:_ Adam | _Loss Function:_ Binary Crossentropy

## Installation & Run

### 1\. Clone the Repository

```bash
git clone [https://github.com/yourusername/brain-tumor-detection.git](https://github.com/yourusername/brain-tumor-detection.git)
cd brain-tumor-detection
```

### 2\. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Run the Application

Make sure `deMLon_model.h5` is in the root directory.

```bash
python app.py
```

### 5\. Access the Web UI

Open your browser and go to:
`http://127.0.0.1:5000/`

## 📊 Training (Optional)

If you want to retrain the model yourself:

1.  Place your dataset in a folder named `Training`.
2.  Run the training script:

<!-- end list -->

```bash
python training_model.py
```

This will generate training graphs and save a new `deMLon_model.h5`.

## 📈 Results

- **Training Accuracy:** \~98% (varies based on epochs)
- **Validation Accuracy:** \~96%
- The system uses a confidence threshold of **0.5** to determine the presence of a tumor.

---

made for Tejas Expo
