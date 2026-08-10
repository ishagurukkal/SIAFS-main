from flask import Flask, render_template, request, send_file, session
import os
import cv2
from werkzeug.utils import secure_filename

# Import your custom modules
from deepfake_detector import detect_deepfake
from photoshop_detector import detect_photoshop
from utils.pdf_report import generate_report

app = Flask(__name__)

# --- CONFIGURATION ---
# Secret key is required for 'session' to work
app.secret_key = "sahyadri_forensic_suite_2026"

# Define Absolute Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")

# Ensure all forensic directories exist
for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    mode = request.form.get('mode')
    file = request.files['image']

    if file.filename == '':
        return "No selected file", 400

    # 1. Secure the filename (important for Windows paths)
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # 2. Perform Analysis based on Mode
    if mode == "deepfake":
        # Run CNN-MobileNet Detection
        result, score, heatmap = detect_deepfake(image_path)
        
        map_filename = "deepfake_heatmap.jpg"
        map_path = os.path.join(HEATMAP_FOLDER, map_filename)
        cv2.imwrite(map_path, heatmap)
        
    else:
        # Run ELA-CASIA Detection
        result, score, ela_map = detect_photoshop(image_path)
        
        map_filename = "ela_analysis.jpg"
        map_path = os.path.join(HEATMAP_FOLDER, map_filename)
        cv2.imwrite(map_path, ela_map)

    # 3. Store results in Session for the PDF Generator
    session['last_scan'] = {
        'image_path': image_path,
        'mode': mode,
        'result': result,
        'score': round(float(score) * 100, 2), # Convert to percentage
        'map_path': map_path,
        'filename': filename
    }

    return render_template(
        "result.html",
        result=result,
        score=score, # Original score for logic
        image="static/uploads/" + filename,
        map="static/heatmaps/" + map_filename,
        type=mode
    )

@app.route('/download')
def download():
    # Retrieve data from the last scan
    data = session.get('last_scan')
    
    if not data:
        return "Forensic data expired or not found. Please re-analyze.", 400

    report_filename = f"Forensic_Report_{data['filename']}.pdf"
    report_path = os.path.join(REPORT_FOLDER, report_filename)

    # Generate the PDF with the SHA-256 Hash (handled inside generate_report)
    generate_report(
        data['image_path'],
        data['mode'],
        data['result'],
        data['score'],
        data['map_path'],
        report_path
    )

    return send_file(report_path, as_attachment=True)

if __name__ == "__main__":
    print("--- SIAFS Server Starting ---")
    print(f"Upload directory: {UPLOAD_FOLDER}")
    app.run(debug=True, port=5000)