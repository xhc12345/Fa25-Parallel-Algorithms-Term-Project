import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import warnings
from model import SimpleCNN

def load_int8_model(model_path, device):
    """
    Loads a saved INT8 quantized model state_dict.
    
    To load a quantized model, you MUST first create an instance
    of the model and then apply the same quantization steps
    (fuse, qconfig, prepare, convert) *before* loading the state_dict.
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
    model.load_state_dict(torch.load(model_path))
    
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
    #    strict=False is needed because the FP32 model was
    #    saved with quant/dequant stubs which we don't
    #    strictly need for FP32 inference, but the class has them.
    model.load_state_dict(torch.load(model_path), strict=False)
    
    model.to(device)
    model.eval() # Set to evaluation mode
    
    print("FP32 model loaded successfully.")
    return model

def preprocess_image(image_path):
    """
    Loads an image and preprocesses it to match the
    MNIST training data (1x28x28, normalized).
    """
    # These are the mean and std dev of the MNIST dataset from your notebook
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Ensure it's 1-channel
        transforms.Resize((28, 28)),                # Resize to 28x28
        transforms.ToTensor(),                      # Convert to tensor (scales to [0, 1])
        transforms.Normalize(mnist_mean, mnist_std) # Normalize
    ])
    
    # Load the image
    image = Image.open(image_path)
    
    # --- IMPORTANT ---
    # MNIST data is black digits on a white background.
    # If your image is a white digit on a black background
    # (like a photo of chalk), you must invert it.
    #
    # Uncomment the line below if your images are white-on-black.
    # from torchvision.transforms.functional import invert
    # image = invert(image)
    # -------------------

    # Apply transforms and add a batch dimension (BCHW)
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

def predict(model, image_tensor, device):
    """Runs inference on a preprocessed image tensor."""
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad(): # Disable gradient calculation
        output = model(image_tensor)
        # Get the index of the highest score
        _, predicted = torch.max(output.data, 1)
        
    return predicted.item()

# --- Main execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # --- Configuration ---
    DEVICE = torch.device("cpu")
    INT8_MODEL_PATH = os.path.join("bin", "CNN-MNIST-CPU-int8.pt")
    FP32_MODEL_PATH = os.path.join("bin", "CNN-MNIST-CPU-fp32.pt")

    IMAGE_TO_TEST = os.path.join("..", "..", "uploads", "original_received.png")
    
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"Error: Image file not found at {IMAGE_TO_TEST}")
        print("Please update IMAGE_TO_TEST with a valid path.")
    elif not os.path.exists(INT8_MODEL_PATH):
        print(f"Error: INT8 Model not found at {INT8_MODEL_PATH}")
        print("Please update INT8_MODEL_PATH with a valid path.")
    elif not os.path.exists(FP32_MODEL_PATH):
        print(f"Error: FP32 Model not found at {FP32_MODEL_PATH}")
        print("Please update FP32_MODEL_PATH with a valid path.")
    else:
        print("Starting prediction...\n")
        # --- 1. Load the Model ---
        # Choose which model to load:
        
        int8_model = load_fp32_model(FP32_MODEL_PATH, DEVICE)
        fp32_model = load_int8_model(INT8_MODEL_PATH, DEVICE)

        # --- 2. Preprocess the Image ---
        print(f"\nProcessing image: {IMAGE_TO_TEST}")
        input_tensor = preprocess_image(IMAGE_TO_TEST)
        
        # --- 3. Get Prediction ---
        int8_pred = predict(int8_model, input_tensor, DEVICE)
        fp32_pred = predict(fp32_model, input_tensor, DEVICE)
        
        print(f"\n----------------------------")
        print(f"FP32 Model prediction: {fp32_pred}")
        print(f"INT8 Model prediction: {int8_pred}")
        print(f"----------------------------")