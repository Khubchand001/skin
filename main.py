from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import timm
import torchvision.transforms as transforms
import cv2
import numpy as np

# -------------------- APP --------------------
app = FastAPI(title="SkinCare AI API")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow Flutter / Web / Mobile
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- CONSTANTS --------------------
NUM_CLASSES = 7
CLASS_NAMES = [
    "Cellulitis",
    "Impetigo",
    "Athlete-foot",
    "Nail-fungus",
    "Ringworms",
    "Cutaneous-larva-migrans",
    "Chickenpox"
]

MODEL_PATH = "skin_disease_rtx3050ti_weights.pth"

# -------------------- MODEL --------------------
print("🔄 Loading model...")

model = timm.create_model(
    "tf_efficientnet_b4.ns_jft_in1k",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)
model.eval()

print("✅ Model loaded successfully")

# -------------------- TRANSFORMS --------------------
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------- SKIN FILTER --------------------
def skin_filter(image: Image.Image) -> bool:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))

    # ---------- 1️⃣ HSV SKIN MASK ----------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = cv2.countNonZero(mask) / mask.size

    print(f"🧪 Skin ratio: {skin_ratio:.3f}")

    # STRICT threshold
    if skin_ratio < 0.18: 
        return False

    # ---------- 2️⃣ TEXTURE CHECK ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    print(f"🧪 Texture variance: {lap_var:.2f}")

    # Reject flat images (charts, cartoons, screenshots)
    if lap_var < 35:
        return False

    # ---------- 3️⃣ EDGE DENSITY CHECK ----------
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = np.sum(edges > 0) / edges.size

    print(f"🧪 Edge ratio: {edge_ratio:.3f}")

    # Skin photos have moderate edges
    if edge_ratio < 0.02 or edge_ratio > 0.25:
        return False

    return True


# -------------------- UTILS --------------------
def get_severity(conf: float) -> str:
    if conf < 0.60:
        return "Mild"
    elif conf < 0.85:
        return "Moderate"
    return "Severe"

# -------------------- ROUTES --------------------
@app.get("/")
def root():
    return {"status": "SkinCare AI API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print("📥 Image received:", image.size)

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

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return {"error": str(e)}
