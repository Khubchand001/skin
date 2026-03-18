# app/predict.py

from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np


# ---------------------------------
# Device
# ---------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------
# Classes
# ---------------------------------
CLASS_NAMES = [
    "Akne",
    "Athlete_foot",
    "Benign",
    "Cellulitis",
    "Chickenpox",
    "Impetigo",
    "Nail-fungus",
    "Pigment",
    "Ringworm",
    "Shingles",
    "Normal_skin"
]


# ---------------------------------
# Transform
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------
# Skin + Blur Filter (Improved)
# ---------------------------------
def skin_filter(image: Image.Image) -> bool:

    img = np.array(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 40, 70])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    skin_ratio = np.sum(mask > 0) / mask.size

    # Skin presence check
    if skin_ratio < 0.07:
        return False

    # Blur detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var < 50:
        return False

    return True


# ---------------------------------
# Test Time Augmentation (TTA)
# ---------------------------------
def predict_tta(model, image_tensor):

    images = [
        image_tensor,
        torch.flip(image_tensor, [3]),
        torch.rot90(image_tensor, 1, [2, 3])
    ]

    outputs = []

    with torch.no_grad():
        for img in images:
            out = model(img)
            out = F.softmax(out, dim=1)
            outputs.append(out)

    return torch.mean(torch.stack(outputs), dim=0)[0]


# ---------------------------------
# Severity Logic
# ---------------------------------
def get_severity(conf: float) -> str:

    if conf < 0.60:
        return "Mild"
    elif conf < 0.85:
        return "Moderate"
    return "Severe"


# ---------------------------------
# Main Prediction Function
# ---------------------------------
def predict_image(model, image: Image.Image):

    # Validate input
    if not skin_filter(image):
        return {
            "disease": "Unknown",
            "confidence": 0,
            "severity": "Unknown",
            "probabilities": [],
            
          
            
        }

    # Preprocess
    x = transform(image).unsqueeze(0).to(DEVICE)

    model = model.to(DEVICE)
    model.eval()

    # TTA Prediction
    probs = predict_tta(model, x)

    confidence, idx = torch.max(probs, dim=0)

    conf_value = float(confidence.item())

    
    return {
        "disease": CLASS_NAMES[idx.item()].replace("_", " "),
        "confidence": round(conf_value * 100, 2),
        "severity": get_severity(conf_value),
        "probabilities": probs.tolist(),
        
      
    }