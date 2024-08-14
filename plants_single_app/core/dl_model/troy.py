import io
import json
import os
import torch
import torch.onnx
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from skorch import NeuralNetClassifier

# Define the path to the directory containing the model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Update paths to be relative to the script's directory
labels_path = os.path.join(BASE_DIR, 'plant_classes.json')
model_params_path = os.path.join(BASE_DIR, 'model_params.pt')
model_optimizer_path = os.path.join(BASE_DIR, 'model_optimizer.pt')
model_history_path = os.path.join(BASE_DIR, 'model_history.json')

# Load labels
try:
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    print("Labels loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading labels: {e}")

class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)

def load_model():
    """Load Pytorch model."""
    global model
    try:
        # Initialize the NeuralNetClassifier
        model = NeuralNetClassifier(
            PretrainedModel,
            criterion=nn.CrossEntropyLoss,
            module__output_features=38
        )
        model.initialize()
        print("NeuralNetClassifier initialized successfully.")
        
        # Load the saved model weights
        model.load_params(
            f_params=model_params_path,
            f_optimizer=model_optimizer_path,
            f_history=model_history_path
        )
        print("Model parameters loaded successfully.")
        
        # Print model status for debugging
        print(f"Model state dict keys: {list(model.module_.state_dict().keys())}")
        
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

def classify_image(image_bytes):
    '''
    Function to pre-process the input image to a format similar to training data and
    then make predictions on the image.
    '''
    # Define the various image transformations we have to make
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Open and identify the uploaded/given image file.
    image = Image.open(io.BytesIO(image_bytes))
    # Pre-process the uploaded image
    tensor = transform(image).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        outputs = model.forward(tensor)
    _, indices = torch.sort(outputs, descending=True)
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Return the top 3 predictions ('class label, probability')
    result = [(labels[str(idx.item())][0], labels[str(idx.item())][1], confidence[idx].item()) for idx in indices[0][:5]]

    return result

def export_model_to_onnx():
    """Export the PyTorch model to ONNX format."""
    # Ensure `model` is defined and accessible
    if 'model' not in globals() or model is None:
        raise RuntimeError("Model is not loaded properly. Ensure load_model() is called successfully.")
    
    # Check if model is an instance of NeuralNetClassifier
    if not isinstance(model, NeuralNetClassifier):
        raise RuntimeError("Model is not an instance of NeuralNetClassifier. Check if `model` is correctly loaded.")
    
    # Extract the underlying PyTorch model from NeuralNetClassifier
    try:
        pytorch_model = model.module_
        print("PyTorch model extracted successfully.")
    except AttributeError:
        raise RuntimeError("Failed to extract the PyTorch model from NeuralNetClassifier. Check if `model` is correctly loaded.")
    
    # Set the model to evaluation mode
    pytorch_model.eval()
    
    # Define dummy input with the same shape as your actual inputs
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on your model input size
    
    # Export the model
    onnx_path = "model.onnx"
    try:
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            verbose=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Model has been exported to {onnx_path}")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")

if __name__ == "__main__":
    # Load the model
    load_model()
    
    # Export the model to ONNX format
    export_model_to_onnx()
