from fpdf import FPDF
import datetime
import hashlib
import os

def calculate_sha256(filepath):
    """Calculates a unique SHA-256 digital fingerprint for the image."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_report(image_path, result_type, result_label, score, map_path, output_path):
    # 1. Generate the hash
    img_hash = calculate_sha256(image_path)
    
    # 2. Setup PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(190, 10, "SIAFS DIGITAL FORENSIC REPORT", ln=True, align='C')
    
    # Digital Fingerprint (Hash) Section
    pdf.ln(5)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 8, " EVIDENCE INTEGRITY: SHA-256 DIGITAL HASH", ln=True, fill=True)
    pdf.set_font("Courier", '', 9) 
    pdf.multi_cell(190, 7, img_hash, border=1, align='C')
    pdf.ln(5)

    # Verdict and Metadata
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(95, 10, f"Analysis Mode: {result_type.upper()}", border=0)
    pdf.cell(95, 10, f"Result: {result_label.upper()}", border=0, ln=True)
    pdf.cell(95, 10, f"Confidence Score: {score}%", border=0)
    pdf.cell(95, 10, f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", border=0, ln=True)
    pdf.ln(10)

    # Images Section
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(95, 5, "Original Evidence", ln=0)
    pdf.cell(95, 5, "Artifact Analysis Map", ln=1)
    
    curr_y = pdf.get_y()
    # Adding Images
    pdf.image(image_path, x=10, y=curr_y + 5, w=90)
    pdf.image(map_path, x=110, y=curr_y + 5, w=90)
    
    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(190, 5, "This report is generated automatically by the SIAFS Forensic Engine.", ln=True, align='C')
    pdf.cell(190, 5, f"Integrity Verified - Case ID: {img_hash[:8].upper()}", ln=True, align='C')

    pdf.output(output_path)