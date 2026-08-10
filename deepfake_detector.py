import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.grad_cam import compute_gradcam, overlay_heatmap

model_path = "models/deepfake_model.h5"
model = load_model(model_path)

def detect_deepfake(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return "Error", 0, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_input = np.reshape(img_resized / 255.0, (1, 128, 128, 3))

    prediction = model.predict(img_input)[0][0]

    # Logic: 0=Fake, 1=Real (Alphabetical: F before R)
    if prediction > 0.6:
        label, confidence = "Real", prediction
    elif prediction < 0.4:
        label, confidence = "Fake", 1 - prediction
    else:
        label, confidence = "Uncertain", prediction

    try:
        # 'out_relu' is the standard last layer for MobileNetV2
        heatmap_raw = compute_gradcam(model, img_input, 'out_relu')
        visual = overlay_heatmap(heatmap_raw, img_bgr)
    except:
        visual = cv2.applyColorMap(cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)

    return label, confidence, visual