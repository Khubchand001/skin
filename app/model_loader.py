# app/model_loader.py

import os
import torch
import timm
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download


NUM_CLASSES = 11   # must match CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(1)  # 🔥 prevent CPU overload


# ---------------------------------
# Load Skin Disease Model (UNCHANGED)
# ---------------------------------
def load_model():

    print("🔄 Downloading model from Hugging Face...")

    model_path = hf_hub_download(
        repo_id="khubchand/skin_model1",
        filename="best_model.pth",
        token=os.getenv("HF_TOKEN")
    )

    print("🔄 Creating model architecture...")

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    print("🔄 Loading weights...")

    state_dict = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully")

    return model


# ---------------------------------
# Load Lightweight Object Model (FAST)
# ---------------------------------
_object_model = None

def load_object_model():
    global _object_model

    if _object_model is None:
        print("🔄 Loading MobileNet for object detection...")
        _object_model = models.mobilenet_v3_small(pretrained=True)
        _object_model.to(DEVICE)
        _object_model.eval()

    return _object_model


# ---------------------------------
# Transform for object detection
# ---------------------------------
object_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------------------
# FAST Skin Detection (HSV)
# ---------------------------------
def is_skin_image(image: Image.Image) -> bool:

    img = np.array(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 40, 70])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    skin_ratio = np.sum(mask > 0) / mask.size

    return skin_ratio > 0.07


# ---------------------------------
# Blur Detection (FAST)
# ---------------------------------
def is_blurry(image: Image.Image) -> bool:

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return lap_var < 50


# ---------------------------------
# Animal Detection (ImageNet-based)
# ---------------------------------
ANIMAL_CLASSES = [
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
]


def is_animal(image: Image.Image):

    model = load_object_model()

    img = object_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    top_prob, top_idx = torch.topk(probs, 1)

    # ImageNet class labels (simplified check)
    predicted_class = str(top_idx.item())

    # 🔥 lightweight heuristic (fast)
    if any(animal in predicted_class for animal in ANIMAL_CLASSES):
        return True

    return False


# ---------------------------------
# Combined Validation (SUPER FAST)
# ---------------------------------
def validate_image(image: Image.Image):

    if is_blurry(image):
        return False, "Image is blurry"

    if is_animal(image):
        return False, "Animal detected, not skin"

    if not is_skin_image(image):
        return False, "No skin detected"

    return True, "Valid skin image"