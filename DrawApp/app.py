import os
import re
import base64
import io
import time # Import time for dummy delays
import glob # To find files
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# --- Configuration ---
app = Flask(__name__)

# --- Constants ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ORIGINAL_FILENAME = 'original_received.png'
PROCESSED_FILENAME = 'processed.png' # Changed from processed_28x28.png

# Map selection keys to final image dimensions
SIZE_MAP = {
    "small": 28,
    "medium": 64,
    "large": 128
}

# --- PyTorch Pre-processing ---

def get_preprocessor(target_size):
    """Returns a pre-processing pipeline for a specific target size."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((target_size, target_size), antialias=True), 
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x) # Invert
    ])

# --- Helper Functions ---

def decode_image_data(image_data_url):
    """Strips header and decodes base64 image."""
    image_data_str = re.sub('^data:image/.+;base64,', '', image_data_url)
    return base64.b64decode(image_data_str)

def preprocess_image(image_bytes, target_size):
    """Opens image bytes, applies pre-processing pipeline, and returns batch tensor."""
    # Get the specific pre-processor for the requested size
    preprocessor = get_preprocessor(target_size)
    
    img = Image.open(io.BytesIO(image_bytes))
    tensor = preprocessor(img)
    batch_tensor = tensor.unsqueeze(0) # Add batch dimension [1, 1, H, W]
    return batch_tensor

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML drawing page."""
    return render_template('digit-drawer.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serves files from the UPLOAD_FOLDER."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/get-images', methods=['GET'])
def get_images():
    """Checks for existing images and returns their URLs."""
    original_path = os.path.join(UPLOAD_FOLDER, ORIGINAL_FILENAME)
    processed_path = os.path.join(UPLOAD_FOLDER, PROCESSED_FILENAME) # Updated name

    images = {}
    if os.path.exists(original_path):
        images['original'] = f'/uploads/{ORIGINAL_FILENAME}?t={os.path.getmtime(original_path)}'
    
    if os.path.exists(processed_path):
        images['processed'] = f'/uploads/{PROCESSED_FILENAME}?t={os.path.getmtime(processed_path)}'
        
    return jsonify(images)

@app.route('/process-image', methods=['POST'])
def process_image_route():
    """Receives drawn image, saves it, and pre-processes it."""
    try:
        data = request.json
        image_bytes = decode_image_data(data['imageData'])
        
        # Get the selected size, default to 'small' and 28
        selected_size_key = data.get('size', 'small')
        target_size = SIZE_MAP.get(selected_size_key, 28)

        # --- Save Original Image ---
        original_save_path = os.path.join(UPLOAD_FOLDER, ORIGINAL_FILENAME)
        img = Image.open(io.BytesIO(image_bytes))
        img.convert('RGB').save(original_save_path)
        
        # --- Pre-process and Save Processed Image ---
        batch_tensor = preprocess_image(image_bytes, target_size)
        processed_save_path = os.path.join(UPLOAD_FOLDER, PROCESSED_FILENAME) # Updated name
        save_image(batch_tensor.squeeze(0), processed_save_path)
        
        print(f"Successfully processed image to size {target_size}x{target_size}. Tensor shape: {batch_tensor.shape}")

        return jsonify({
            'status': 'success',
            'message': f'Image processed to {target_size}x{target_size} and saved!',
            'original_path': original_save_path,
            'processed_path': processed_save_path
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# --- Prediction Endpoints (Now read from disk) ---

def run_prediction(model_name, model_logic):
    """Helper to run prediction logic for both models."""
    try:
        data = request.json
        # Get the selected size, default to 'small' and 28
        selected_size_key = data.get('size', 'small')
        target_size = SIZE_MAP.get(selected_size_key, 28)

        image_path = os.path.join(UPLOAD_FOLDER, ORIGINAL_FILENAME)
        if not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': 'Original image not found. Please draw and send one.'}), 404

        # Read and process the *saved* original image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        batch_tensor = preprocess_image(image_bytes, target_size)
        
        print(f"{model_name} processing tensor of shape: {batch_tensor.shape}")
        
        # Run the specific dummy logic
        result = model_logic(batch_tensor)
        return jsonify(result)

    except Exception as e:
        print(f"Error in {model_name}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict/model_one', methods=['POST'])
def predict_model_one():
    """Dummy Endpoint 1: Simulates a Digit Classifier."""
    def logic(batch_tensor):
        time.sleep(1.0) # Simulate processing
        prediction = torch.randint(0, 10, (1,)).item()
        confidence = torch.rand((1,)).item() * 0.3 + 0.7
        return {
            'model_name': 'Digit Classifier',
            'prediction': f"Digit {prediction}",
            'confidence': f"{confidence:.4f}"
        }
    return run_prediction('Model 1', logic)

@app.route('/predict/model_two', methods=['POST'])
def predict_model_two():
    """Dummy Endpoint 2: Simulates a Letter Classifier."""
    def logic(batch_tensor):
        time.sleep(0.5) # Simulate processing
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        prediction_index = torch.randint(0, 26, (1,)).item()
        prediction = letters[prediction_index]
        confidence = torch.rand((1,)).item() * 0.4 + 0.5
        return {
            'model_name': 'Letter Classifier',
            'prediction': f"Letter {prediction}",
            'confidence': f"{confidence:.4f}"
        }
    return run_prediction('Model 2', logic)


if __name__ == '__main__':
    app.run(debug=True, port=5000)