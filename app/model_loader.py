```python
# app/model_loader.py

import torch
import timm
from huggingface_hub import hf_hub_download

NUM_CLASSES = 11   # must match CLASS_NAMES

def load_model():

    print("🔄 Downloading model from Hugging Face...")

    model_path = hf_hub_download(
        repo_id="khubchand/skin_model1",
        filename="best_model.pth"
    )

    print("🔄 Creating model architecture...")

    model = timm.create_model(
        "tf_efficientnet_b4.ns_jft_in1k",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    print("🔄 Loading weights...")

    state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict)

    model.eval()

    print("✅ Model loaded successfully")

    return model
```
