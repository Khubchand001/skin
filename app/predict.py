# app/predict.py

from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

CLASS_NAMES = [
    "Cellulitis",
    "Impetigo",
    "Athlete-foot",
    "Nail-fungus",
    "Ringworms",
    "Cutaneous-larva-migrans",
    "Chickenpox"
]

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def skin_filter(image: Image.Image) -> bool:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = cv2.countNonZero(mask) / mask.size

    if skin_ratio < 0.18:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var < 35:
        return False

    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = np.sum(edges > 0) / edges.size

    if edge_ratio < 0.02 or edge_ratio > 0.25:
        return False

    return True


def get_severity(conf: float) -> str:
    if conf < 0.60:
        return "Mild"
    elif conf < 0.85:
        return "Moderate"
    return "Severe"


def predict_image(model, image: Image.Image):
    if not skin_filter(image):
        return {"error": "Uploaded image is not a skin image"}

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)

    disease = CLASS_NAMES[idx.item()]
    conf_value = float(confidence.item())

    return {
        "disease": disease,
        "confidence": round(conf_value * 100, 2),
        "severity": get_severity(conf_value),
        "probabilities": probs.tolist(),
    }
