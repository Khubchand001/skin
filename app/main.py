# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.model_loader import load_model
from app.predict import predict_image

app = FastAPI(title="SkinCare AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting API...")
model = None

def get_model():
    global model
    if model is None:
        print("🔄 Loading model for first request...")
        model = load_model()
    return model

@app.get("/")
def root():
    return {"status": "SkinCare AI API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        model_instance = get_model()  # 🔥 Lazy load here

        result = predict_image(model_instance, image)

        return result

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return {"error": str(e)}