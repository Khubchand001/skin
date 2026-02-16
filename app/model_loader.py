# app/model_loader.py

import torch
import timm
from huggingface_hub import hf_hub_download

NUM_CLASSES = 7

def load_model():
    print("🔄 Downloading model from Hugging Face...")

    model_path = hf_hub_download(
        repo_id="khubchand/skin-model",   
        filename="skin_disease_rtx3050ti_weights.pth"
    )

    print("🔄 Creating model architecture...")

    model = timm.create_model(
        "tf_efficientnet_b4.ns_jft_in1k",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    print("🔄 Loading weights...")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    model.eval()
    print("✅ Model loaded successfully")

    return model
