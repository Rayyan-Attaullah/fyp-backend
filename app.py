from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model architecture
model = models.resnet50(weights='DEFAULT')  # Load ResNet50 model with pretrained weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Adjusting the final layer for binary classification

# Load the model weights
model.load_state_dict(torch.load('cancer-model.pth', map_location=device))  # Load saved model weights
model = model.to(device)  # Move model to the appropriate device
model.eval()  # Set model to evaluation mode

# Define the image preprocessing function
def preprocess_image(image_bytes):
    # Image preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    
    # Convert the image bytes to PIL Image and apply preprocessing
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
    return image.to(device)  # Ensure image is on the same device as the model

# Define the endpoint to handle image upload and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image
        image = preprocess_image(file.read())

        # Run the model prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Map predicted class to cancer type (adjust based on your model)
        cancer_classes = ['benign', 'malignant']
        result = cancer_classes[predicted_class]

        return jsonify({
            'prediction': result,
            'probabilities': probabilities.tolist()[0]  # Return the probabilities for both classes
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
