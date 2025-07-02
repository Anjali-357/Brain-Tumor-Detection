import os
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, make_response, jsonify, abort
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import qrcode
import urllib.parse
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = './uploads'
REPORTS_FOLDER = './reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create directories
for folder in [UPLOAD_FOLDER, REPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Enhanced model configuration
os.environ["TF_USE_LEGACY_KERAS"] = "1"
MODEL_PATH = 'models/model1.keras'
CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Enhanced descriptions with medical details
TUMOR_DESCRIPTIONS = {
    "notumor": {
        "description": "No signs of a brain tumor were detected in the uploaded scan. The analyzed image shows normal brain tissue patterns without abnormal growths or lesions.",
        "recommendation": "Continue regular health check-ups. Consult a neurologist if symptoms persist.",
        "severity": "Normal"
    },
    "glioma": {
        "description": "Glioma tumors develop from glial cells that support nerve cells in the brain. These tumors can vary significantly in their growth rate and severity.",
        "recommendation": "Immediate consultation with a neuro-oncologist is recommended. Further imaging and biopsy may be required.",
        "severity": "High Priority"
    },
    "meningioma": {
        "description": "Meningioma tumors arise from the meninges, the protective layers surrounding the brain and spinal cord. Most are benign and slow-growing.",
        "recommendation": "Regular monitoring with MRI scans. Treatment options include observation, surgery, or radiation therapy.",
        "severity": "Moderate Priority"
    },
    "pituitary": {
        "description": "Pituitary tumors develop in the pituitary gland, which controls hormone production. Most pituitary adenomas are benign and treatable.",
        "recommendation": "Endocrinologist consultation recommended. Hormone level testing and treatment planning required.",
        "severity": "Moderate Priority"
    }
}

# Global model variable
model = None

def load_detection_model():
    """Load the tumor detection model with proper error handling"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            logger.info("Tumor detection model loaded successfully")
            return True
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        logger.critical(f"Critical error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """Enhanced image preprocessing with validation"""
    try:
        # Validate image file
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Load and preprocess for model
            processed_img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(processed_img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

def predict_tumor(image_path):
    """Enhanced tumor prediction with comprehensive error handling"""
    if model is None:
        logger.error("Model not loaded for prediction")
        return None, 0.0, None, "Model not available"
    
    try:
        img_array = preprocess_image(image_path)
        
        # Get model predictions
        predictions = model.predict(img_array, verbose=0)
        confidence_scores = predictions[0]
        
        # Get primary prediction
        primary_index = np.argmax(confidence_scores)
        primary_confidence = float(confidence_scores[primary_index])
        primary_label = CLASS_LABELS[primary_index]
        
        # Get all predictions for detailed analysis
        all_predictions = {
            CLASS_LABELS[i]: float(confidence_scores[i]) 
            for i in range(len(CLASS_LABELS))
        }
        
        # Format result
        if primary_label == 'notumor':
            result_text = "No Tumor Detected"
        else:
            result_text = f"Tumor Type: {primary_label.capitalize()}"
        
        tumor_info = TUMOR_DESCRIPTIONS[primary_label]
        
        return result_text, primary_confidence, tumor_info, all_predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, 0.0, None, f"Prediction failed: {str(e)}"

class EnhancedPDF(FPDF):
    """Enhanced PDF class with better styling and medical report formatting"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Professional header with gradient-like effect"""
        # Blue header background
        self.set_fill_color(30, 64, 175)  # Professional blue
        self.rect(0, 0, 210, 25, 'F')
        
        # White text for header
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 16)
        self.set_xy(10, 8)
        self.cell(0, 10, 'MRI TUMOR DETECTION REPORT', ln=True, align='C')
        
        # Subtitle
        self.set_font('Arial', '', 10)
        self.set_xy(10, 16)
        self.cell(0, 6, 'Powered by AI-Assisted Medical Imaging Analysis', ln=True, align='C')
        
        # Reset text color
        self.set_text_color(0, 0, 0)
    
    def footer(self):
        """Professional footer with page numbers and disclaimer"""
        self.set_y(-20)
        
        # Disclaimer line
        self.set_font('Arial', 'I', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, 'This report is generated by AI analysis and should be reviewed by qualified medical professionals.', ln=True, align='C')
        
        # Page number
        self.set_font('Arial', '', 9)
        self.cell(0, 8, f'Page {self.page_no()}', align='C')
        
        # Reset text color
        self.set_text_color(0, 0, 0)
    
    def add_section_title(self, title, y_position=None):
        """Add styled section title"""
        if y_position:
            self.set_y(y_position)
        
        self.set_fill_color(240, 248, 255)  # Light blue background
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, ln=True, fill=True, border=1)
        self.ln(2)
    
    def add_info_box(self, content, bg_color=(245, 245, 245), text_color=(0, 0, 0)):
        """Add styled information box"""
        self.set_fill_color(*bg_color)
        self.set_text_color(*text_color)
        self.set_font('Arial', '', 10)
        
        # Calculate height needed
        lines = content.split('\n')
        height = len(lines) * 5 + 4
        
        self.multi_cell(0, 5, content, border=1, fill=True)
        self.ln(3)
        
        # Reset colors
        self.set_text_color(0, 0, 0)

def generate_verification_qr(report_id):
    """Generate QR code for report verification"""
    try:
        # In production, this would be your actual verification URL
        verification_url = f"https://your-domain.com/verify/{report_id}"
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(verification_url)
        qr.make(fit=True)
        
        qr_path = os.path.join(UPLOAD_FOLDER, f'qr_{report_id}.png')
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(qr_path)
        
        return qr_path
    except Exception as e:
        logger.error(f"QR code generation failed: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    """Enhanced main route with better error handling"""
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            
            if not file or not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF files.'}), 400
            
            # Generate unique filename
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            file.save(file_path)
            logger.info(f"File saved: {unique_filename}")
            
            # Perform prediction
            result, confidence, tumor_info, all_predictions = predict_tumor(file_path)
            
            if result is None:
                return jsonify({'error': tumor_info}), 500
            
            return render_template('index.html',
                                 result=result,
                                 confidence=f"{confidence*100:.2f}",
                                 file_path=f"/uploads/{unique_filename}",
                                 description=tumor_info['description'],
                                 recommendation=tumor_info['recommendation'],
                                 severity=tumor_info['severity'],
                                 all_predictions=all_predictions)
                                 
        except Exception as e:
            logger.error(f"Upload processing error: {e}")
            return jsonify({'error': 'Processing failed. Please try again.'}), 500
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded files securely"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        abort(404)

@app.route('/report')
def generate_report():
    """Enhanced report generation with comprehensive PDF styling"""
    try:
        # Get parameters
        result = request.args.get('result', '')
        confidence = request.args.get('confidence', '')
        description = urllib.parse.unquote(request.args.get('description', ''))
        recommendation = urllib.parse.unquote(request.args.get('recommendation', ''))
        severity = urllib.parse.unquote(request.args.get('severity', ''))
        file_path = request.args.get('file_path', '')
        
        if not all([result, confidence, description]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Generate unique report ID
        report_id = uuid.uuid4().hex[:12]
        
        # Get image path
        image_filename = os.path.basename(file_path)
        full_img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        # Generate QR code
        qr_path = generate_verification_qr(report_id)
        
        # Create enhanced PDF
        pdf = EnhancedPDF()
        pdf.add_page()
        
        # Hospital Information Section
        pdf.add_section_title("MEDICAL FACILITY INFORMATION", 35)
        
        hospital_info = (
            "Hospital: NeuroCare Diagnostics Center\n"
            "Attending Physician: Dr. A.K. Verma, MD (Radiology)\n"
            "Address: 123 Health Blvd, Medicity, India\n"
            "Contact: +91-9876543210\n"
            f"Report Generated: {datetime.now().strftime('%d %B %Y at %H:%M')} IST\n"
            f"Report ID: {report_id}"
        )
        pdf.add_info_box(hospital_info, bg_color=(230, 240, 255))
        
        # Patient Information Section
        pdf.add_section_title("PATIENT INFORMATION")
        
        patient_info = (
            "Patient Name: ______________________________\n"
            "Age: ________  Sex: ________  DOB: __________\n"
            "Referring Physician: _______________________\n"
            "Study Date: ________________________________\n"
            "Medical Record Number: ____________________"
        )
        pdf.add_info_box(patient_info, bg_color=(255, 255, 230))
        
        # Detection Results Section
        pdf.add_section_title("AI ANALYSIS RESULTS")
        
        # Determine color based on severity
        if severity == "Normal":
            result_color = (220, 255, 220)  # Light green
        elif severity == "High Priority":
            result_color = (255, 220, 220)  # Light red
        else:
            result_color = (255, 245, 220)  # Light orange
        
        result_info = (
            f"FINDING: {result}\n"
            f"CONFIDENCE LEVEL: {confidence}%\n"
            f"CLINICAL PRIORITY: {severity}\n"
            f"Analysis Algorithm: Deep Learning CNN Model v2.1"
        )
        pdf.add_info_box(result_info, bg_color=result_color)
        
        # Clinical Description Section
        pdf.add_section_title("CLINICAL INTERPRETATION")
        pdf.add_info_box(description)
        
        # Recommendations Section
        pdf.add_section_title("CLINICAL RECOMMENDATIONS")
        pdf.add_info_box(recommendation, bg_color=(240, 255, 240))
        
        # Add images if available
        current_y = pdf.get_y()
        
        # Add scan image
        if os.path.exists(full_img_path):
            try:
                pdf.add_section_title("ANALYZED SCAN IMAGE", current_y + 5)
                pdf.image(full_img_path, x=10, y=pdf.get_y(), w=90, h=60)
            except Exception as e:
                logger.error(f"Failed to add scan image: {e}")
                pdf.add_info_box("Scan image could not be embedded in report")
        
        # Add QR code
        if qr_path and os.path.exists(qr_path):
            try:
                pdf.set_xy(110, current_y + 25)
                pdf.add_section_title("REPORT VERIFICATION")
                pdf.image(qr_path, x=110, y=pdf.get_y(), w=40)
                pdf.set_xy(110, pdf.get_y() + 45)
                pdf.set_font('Arial', '', 9)
                pdf.multi_cell(80, 4, "Scan QR code to verify report authenticity and access digital copy")
            except Exception as e:
                logger.error(f"Failed to add QR code: {e}")
        
        # Add disclaimer and signature section
        pdf.ln(20)
        pdf.add_section_title("AUTHORIZATION & DISCLAIMER")
        
        disclaimer = (
            "IMPORTANT DISCLAIMER: This report is generated using AI-assisted analysis and is intended "
            "to support clinical decision-making. It should not replace professional medical judgment. "
            "All findings must be correlated with clinical symptoms and confirmed by qualified radiologists "
            "or physicians before making treatment decisions.\n\n"
            "Digital Signature: Dr. A.K. Verma, MD\n"
            "License Number: MED12345\n"
            "Date: " + datetime.now().strftime('%d %B %Y')
        )
        pdf.add_info_box(disclaimer, bg_color=(248, 248, 248))
        
        # Generate response
        pdf_output = pdf.output(dest='S').encode('latin1', 'ignore')
        
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Tumor_Detection_Report_{report_id}.pdf'
        
        logger.info(f"Report generated successfully: {report_id}")
        return response
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'error': 'Report generation failed'}), 500

@app.route('/api/health')
def health_check():
    """API endpoint for health checking"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize the application
if __name__ == '__main__':
    # Load model on startup
    model_loaded = load_detection_model()
    if not model_loaded:
        logger.warning("Starting application without model - predictions will not work")
    
    # Start the application
    logger.info("Starting Tumor Detection Application")
    app.run(host='0.0.0.0', port=5050, debug=True)