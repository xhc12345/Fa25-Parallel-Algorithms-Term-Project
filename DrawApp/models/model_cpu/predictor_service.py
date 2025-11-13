import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")
import time # Import time
from model import SimpleCNN 

# --- Model Loading ---

def load_int8_model(model_path, device):
    """
    Loads a saved INT8 quantized model state_dict.
    """
    print(f"Loading quantized INT8 model from: {model_path}")
    
    # 1. Create a new model instance
    model = SimpleCNN().to(device)
    model.eval() # Set to evaluation mode
    
    # 2. Fuse modules (must match the saved model)
    model.fuse_model()
    
    # 3. Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 4. Prepare and convert the model
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    
    # 5. Now, load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    print("INT8 model loaded successfully.")
    return model

def load_fp32_model(model_path, device):
    """Loads a standard FP32 model state_dict."""
    print(f"Loading FP32 model from: {model_path}")
    
    # 1. Create a new model instance
    model = SimpleCNN().to(device)
    
    # 2. Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    model.to(device)
    model.eval() # Set to evaluation mode
    
    print("FP32 model loaded successfully.")
    return model

# --- Pre-processing ---

def preprocess_image(image_path):
    """
    Loads an image and preprocesses it to match the
    MNIST training data (1x28x28, inverted, normalized).
    """
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((28, 28), antialias=True), 
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x), # Invert
        transforms.Normalize(mnist_mean, mnist_std) # Normalize
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

# --- Prediction ---

def get_prediction(model, image_path, device):
    """
    Runs inference on a single image file.
    Returns the predicted class, confidence score, and inference time (ms).
    """
    
    # --- Start Timer ---
    # Times both pre-processing and inference
    start_time = time.perf_counter()
    
    # 1. Preprocess the image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 2. Run prediction
    with torch.no_grad(): 
        output = model(image_tensor)
        
        # 3. Get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get the top class (prediction) and its confidence
        confidence, predicted_index = torch.max(probabilities, 1)
        
        prediction = predicted_index.item()
        confidence_score = confidence.item()

    # --- End Timer ---
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000
        
    return prediction, confidence_score, inference_time_ms