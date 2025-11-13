import os
import re
import base64
import io
import time # Still needed for /process-image, etc.
import glob
import sys 
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# --- Add Model Directory to Python Path ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'model_cpu')
sys.path.append(MODEL_DIR)

# --- Import Model Service ---
try:
    # get_prediction now returns 3 values
    from predictor_service import load_int8_model, load_fp32_model, get_prediction 
except ImportError as e:
    print(f"Error: Failed to import predictor_service.")
    print(f"Make sure {os.path.join(MODEL_DIR, 'predictor_service.py')} exists.")
    print(f"Also, ensure {os.path.join(MODEL_DIR, 'model.py')} exists.")
    print(f"Full error: {e}")
    sys.exit(1)


# --- Configuration ---
app = Flask(__name__)
app.config['MODEL_DIR'] = MODEL_DIR 

# --- Constants ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ORIGINAL_FILENAME = 'original_received.png'
PROCESSED_FILENAME = 'processed.png'

SIZE_MAP = {
    "small": 28,
    "medium": 64,
    "large": 128
}

# --- Load Models on Startup ---
print("Booting server, loading models...")
DEVICE = torch.device("cpu")

INT8_MODEL_PATH = os.path.join(app.config['MODEL_DIR'], "bin", "CNN-MNIST-CPU-int8.pt")
FP32_MODEL_PATH = os.path.join(app.config['MODEL_DIR'], "bin", "CNN-MNIST-CPU-fp32.pt")

# Load FP32 Model (Model 1)
if not os.path.exists(FP32_MODEL_PATH):
    print(f"FATAL ERROR: FP32 Model file not found at {FP32_MODEL_PATH}")
    GLOBAL_MODEL_FP32 = None
else:
    GLOBAL_MODEL_FP32 = load_fp32_model(FP32_MODEL_PATH, DEVICE)
    print("--- FP32 model loading complete. ---")

# Load INT8 Model (Model 2)
if not os.path.exists(INT8_MODEL_PATH):
    print(f"FATAL ERROR: INT8 Model file not found at {INT8_MODEL_PATH}")
    GLOBAL_MODEL_INT8 = None
else:
    GLOBAL_MODEL_INT8 = load_int8_model(INT8_MODEL_PATH, DEVICE)
    print("--- INT8 model loading complete. ---")

print("--- All models loaded. Server ready. ---")


# --- PyTorch Pre-processing (for /process-image route) ---
def get_preprocessor(target_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((target_size, target_size), antialias=True), 
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x)
    ])

# --- Helper Functions ---
def decode_image_data(image_data_url):
    image_data_str = re.sub('^data:image/.+;base64,', '', image_data_url)
    return base64.b64decode(image_data_str)

def preprocess_display_image(image_bytes, target_size):
    preprocessor = get_preprocessor(target_size)
    img = Image.open(io.BytesIO(image_bytes))
    tensor = preprocessor(img)
    batch_tensor = tensor.unsqueeze(0) 
    return batch_tensor

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('digit-drawer.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/get-images', methods=['GET'])
def get_images():
    original_path = os.path.join(UPLOAD_FOLDER, ORIGINAL_FILENAME)
    processed_path = os.path.join(UPLOAD_FOLDER, PROCESSED_FILENAME)

    images = {}
    if os.path.exists(original_path):
        images['original'] = f'/uploads/{ORIGINAL_FILENAME}?t={os.path.getmtime(original_path)}'
    
    if os.path.exists(processed_path):
        images['processed'] = f'/uploads/{PROCESSED_FILENAME}?t={os.path.getmtime(processed_path)}'
        
    return jsonify(images)

@app.route('/process-image', methods=['POST'])
def process_image_route():
    try:
        data = request.json
        image_bytes = decode_image_data(data['imageData'])
        
        selected_size_key = data.get('size', 'small')
        target_size = SIZE_MAP.get(selected_size_key, 28)

        original_save_path = os.path.join(UPLOAD_FOLDER, ORIGINAL_FILENAME)
        img = Image.open(io.BytesIO(image_bytes))
        img.convert('RGB').save(original_save_path)
        
        batch_tensor = preprocess_display_image(image_bytes, target_size)
        processed_save_path = os.path.join(UPLOAD_FOLDER, PROCESSED_FILENAME)
        save_image(batch_tensor.squeeze(0), processed_save_path)
        
        print(f"Successfully processed display image to size {target_size}x{target_size}.")

        return jsonify({
            'status': 'success',
            'message': f'Image processed to {target_size}x{target_size} and saved!',
            'original_path': original_save_path,
            'processed_path': processed_save_path
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --- Prediction Endpoints (Simplified) ---

@app.route('/predict/model_one', methods=['POST'])
def predict_model_one():
    """
    Runs prediction using the REAL FP32 model.
    """
    try:
        if GLOBAL_MODEL_FP32 is None:
            return jsonify({'status': 'error', 'message': 'FP32 Model not loaded.'}), 500
            
        image_path = os.path.join(UPLOAD_FOLDER, ORIGINAL_FILENAME)
        if not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': 'Original image not found.'}), 404

        print(f"Model 1 (FP32) running prediction on: {image_path}")
        
        # --- Call function and unpack all 3 values ---
        prediction, confidence, inf_time = get_prediction(
            GLOBAL_MODEL_FP32, image_path, DEVICE
        )
        
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Time: {inf_time:.2f} ms")

        return jsonify({
            'model_name': 'Digit Classifier (FP32)',
            'prediction': f"Digit {prediction}",
            'confidence': f"{confidence:.4f}",
            'inference_time_ms': f"{inf_time:.2f}" 
        })

    except Exception as e:
        print(f"Error in Model 1 (FP32): {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict/model_two', methods=['POST'])
def predict_model_two():
    """
    Runs prediction using the REAL INT8 model.
    """
    try:
        if GLOBAL_MODEL_INT8 is None:
            return jsonify({'status': 'error', 'message': 'INT8 Model not loaded.'}), 500
            
        image_path = os.path.join(UPLOAD_FOLDER, ORIGINAL_FILENAME)
        if not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': 'Original image not found.'}), 404

        print(f"Model 2 (INT8) running prediction on: {image_path}")
        
        # --- Call function and unpack all 3 values ---
        prediction, confidence, inf_time = get_prediction(
            GLOBAL_MODEL_INT8, image_path, DEVICE
        )
        
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Time: {inf_time:.2f} ms")

        return jsonify({
            'model_name': 'Digit Classifier (INT8)',
            'prediction': f"Digit {prediction}",
            'confidence': f"{confidence:.4f}",
            'inference_time_ms': f"{inf_time:.2f}"
        })
        
    except Exception as e:
        print(f"Error in Model 2 (INT8): {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)