import os
from PIL import Image, ImageChops, ImageEnhance

SOURCE = "dataset/photoshop"
DEST = "dataset/photoshop_ela"

os.makedirs(DEST + "/authentic", exist_ok=True)
os.makedirs(DEST + "/tampered", exist_ok=True)

def convert_ela(image_path):

    try:
        original = Image.open(image_path).convert("RGB")
    except:
        return None

    temp_path = "temp.jpg"

    try:
        original.save(temp_path, "JPEG", quality=90)
        compressed = Image.open(temp_path)

        ela = ImageChops.difference(original, compressed)

        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema])

        if max_diff == 0:
            max_diff = 1

        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)

        return ela

    except:
        return None


for label in ["authentic", "tampered"]:

    folder = os.path.join(SOURCE, label)

    for img in os.listdir(folder):

        path = os.path.join(folder, img)

        ela = convert_ela(path)

        if ela is not None:
            save_path = os.path.join(DEST, label, img)
            ela.save(save_path)

# delete temp file
if os.path.exists("temp.jpg"):
    os.remove("temp.jpg")

print("✅ CLEAN ELA dataset created")