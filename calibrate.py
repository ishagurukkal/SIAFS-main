import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/deepfake_model.h5"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    exit(1)

model = load_model(MODEL_PATH)
target_shape = model.input_shape[1:3]

def evaluate_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    dim = min(h, w)
    
    # Upper-center face-weighted crop
    cy, cx = int(h * 0.42), int(w * 0.50)
    box = int(dim * 0.38)
    crop = img[max(0, cy - box):min(h, cy + box), max(0, cx - box):min(w, cx + box)]
    if crop.size == 0:
        crop = img
        
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_shape, interpolation=cv2.INTER_AREA)
    tensor = np.expand_dims(preprocess_input(resized.astype(np.float32)), axis=0)
    
    return float(model.predict(tensor, verbose=0)[0][0])

# Locate files to audit
target_dir = os.path.abspath("static/uploads")
print(f"--- Scanning Target Directory: {target_dir} ---")

if not os.path.exists(target_dir):
    print("Directory static/uploads does not exist yet. Upload 2-3 images via your web UI first, or place sample images in a folder.")
else:
    files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("No images found in static/uploads. Upload a real image and a fake image through the app first.")
    else:
        for fname in files:
            full_path = os.path.join(target_dir, fname)
            score = evaluate_image(full_path)
            if score is not None:
                print(f"File: {fname} | Raw Score: {score:.4f}")