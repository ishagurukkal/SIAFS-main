import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
from utils.grad_cam import compute_gradcam, overlay_heatmap

model = load_model("models/photoshop_model.h5")

def get_ela(path):
    original = Image.open(path).convert('RGB')
    temp = "temp_predict.jpg"
    original.save(temp, 'JPEG', quality=90)
    ela = ImageChops.difference(original, Image.open(temp))
    scale = 255.0 / (max([ex[1] for ex in ela.getextrema()]) + 1e-10)
    return ImageEnhance.Brightness(ela).enhance(scale * 1.5)

def detect_photoshop(image_path):
    ela_pil = get_ela(image_path).resize((128, 128))
    img_input = np.reshape(np.array(ela_pil) / 255.0, (1, 128, 128, 3))

    prediction = model.predict(img_input)[0][0]

    # Logic: 0=Authentic, 1=Tampered
    if prediction > 0.6:
        label, confidence = "Tampered", prediction
    elif prediction < 0.4:
        label, confidence = "Authentic", 1 - prediction
    else:
        label, confidence = "Uncertain", prediction

    ela_bgr = cv2.cvtColor(np.array(ela_pil), cv2.COLOR_RGB2BGR)
    try:
        # Replace 'conv2d_2' with your actual last conv layer name from model.summary()
        heatmap_raw = compute_gradcam(model, img_input, 'conv2d_2')
        visual = overlay_heatmap(heatmap_raw, ela_bgr)
    except:
        visual = cv2.applyColorMap(cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

    return label, confidence, visual