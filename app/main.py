
# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.model_loader import load_model
from app.predict import predict_image


app = FastAPI(title="SkinCare AI API")


# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("🚀 Starting API...")

model = None


# ---------------- Model Loader ----------------
def get_model():

    global model

    if model is None:

        print("🔄 Loading model for first request...")

        model = load_model()

    return model


# ---------------- Root ----------------
@app.get("/")
def root():

    return {"status": "SkinCare AI API running"}


# ---------------- Prediction ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # read image bytes
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded")

        # convert to PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # load model
        model_instance = get_model()

        # run prediction
        result = predict_image(model_instance, image)

        # ensure valid response
        if "disease" not in result:
            raise HTTPException(
                status_code=400,
                detail="Prediction failed"
            )

        return result

    except HTTPException as e:
        raise e

    except Exception as e:

        print("🔥 ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

