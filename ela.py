import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def generate_ela(image_path, quality=90):

    original = Image.open(image_path).convert('RGB')

    temp_path = "temp.jpg"
    original.save(temp_path, 'JPEG', quality=quality)

    compressed = Image.open(temp_path)

    ela_image = ImageChops.difference(original, compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_array = np.array(ela_image)

    return ela_array