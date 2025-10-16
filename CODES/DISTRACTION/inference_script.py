# Distracted Driver Detection - Inference Script
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    # Initialize model architecture
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get additional info
    class_names = checkpoint.get('class_names', {})

    return model, class_names

def predict_image(model, image_path, device='cpu'):
    """Predict the class of a single image"""
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
        confidence = probs[0, pred_class].item()

    return pred_class, confidence, probs[0].cpu().numpy()

# Example usage:
if __name__ == "__main__":
    # Load model
    model_path = "distracted_driver_vgg16_best.pth"
    model, class_names = load_model(model_path)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Predict on image
    image_path = "test_image.jpg"
    pred_class, confidence, all_probs = predict_image(model, image_path, device)

    print(f"Predicted class: {pred_class} - {class_names.get(pred_class, 'Unknown')}")
    print(f"Confidence: {confidence:.2%}")
