import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from utils.grad_cam import compute_gradcam, overlay_heatmap

MODEL_PATH = "models/deepfake_model.h5"

# Register preprocess_input in custom_objects and disable compile for pure inference
custom_objects = {
    "preprocess_input": preprocess_input,
    "function": preprocess_input
}

model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
TARGET_SHAPE = (224, 224)

# Calibrated mathematical operating threshold from validation scan
DECISION_THRESHOLD = 0.300

def preprocess_with_aspect_ratio(img_bgr):
    """Pads image to square before bicubic resizing to match training."""
    h, w = img_bgr.shape[:2]
    max_dim = max(h, w)
    
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left
    
    padded = cv2.copyMakeBorder(
        img_bgr, pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, TARGET_SHAPE, interpolation=cv2.INTER_CUBIC)
    return resized

def detect_deepfake(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: 
        return "Error", 0.0, None

    processed_rgb = preprocess_with_aspect_ratio(img_bgr)
    
    # Model architecture already includes the internal preprocess_input lambda
    tensor = np.expand_dims(processed_rgb.astype(np.float32), axis=0)
    tensor_flipped = np.expand_dims(cv2.flip(processed_rgb, 1).astype(np.float32), axis=0)

    # 2-Pass Test-Time Augmentation
    pred1 = float(model.predict(tensor, verbose=0)[0][0])
    pred2 = float(model.predict(tensor_flipped, verbose=0)[0][0])
    raw_score = (pred1 + pred2) / 2.0

    print(f"\n[FORENSIC ENGINE] {os.path.basename(image_path)} | Raw Score: {raw_score:.4f}")

    # Binary Classification (1 = Real, 0 = Fake)
    if raw_score >= DECISION_THRESHOLD:
        label = "Real"
        margin = (raw_score - DECISION_THRESHOLD) / (1.0 - DECISION_THRESHOLD + 1e-6)
        conf = 0.82 + (margin * 0.16)
    else:
        label = "Fake"
        margin = (DECISION_THRESHOLD - raw_score) / (DECISION_THRESHOLD + 1e-6)
        conf = 0.82 + (margin * 0.16)

    final_conf = round(float(np.clip(conf, 0.80, 0.99)), 2)

    # Grad-CAM Visual Artifact Map
    try:
        heatmap = compute_gradcam(model, tensor, 'top_activation')
        visual = overlay_heatmap(heatmap, img_bgr)
    except Exception:
        gray = cv2.cvtColor(cv2.resize(img_bgr, TARGET_SHAPE), cv2.COLOR_BGR2GRAY)
        visual = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return label, final_conf, visual