import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

DROPOUT_RATE = 0.5

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # --- NEW: Add QuantStub ---
        # This will convert the input from float to a quantized type
        self.quant = QuantStub()
        
        # Convolutional Block 1
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        # After conv1: 32x28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: 32x14x14

        # Convolutional Block 2
        # Input: 32x14x14
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        # After conv2: 64x14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: 64x7x7
        
        # Classifier (Fully Connected Layers)
        self.flatten = nn.Flatten()
        # Input size to FC layer: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
        # --- NEW: Add DeQuantStub ---
        # This will convert the output from a quantized type back to float
        self.dequant = DeQuantStub()

    def forward(self, x):
        # --- NEW: Apply quantization ---
        x = self.quant(x)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        # Note: Dropout is automatically disabled by model.eval()
        x = self.dropout(x) 
        x = self.fc2(x) 
        
        # --- NEW: Apply de-quantization ---
        x = self.dequant(x)
        
        return x

    # --- NEW: Method to fuse modules ---
    def fuse_model(self):
        """Fuses Conv/ReLU and Linear/ReLU modules."""
        # Note: The model must be in eval() mode to do this
        assert not self.training, "Model must be in eval() mode to fuse"
        
        # Fuse Conv + ReLU
        torch.quantization.fuse_modules(self, ['conv1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'relu2'], inplace=True)
        # Fuse Linear + ReLU
        torch.quantization.fuse_modules(self, ['fc1', 'relu3'], inplace=True)