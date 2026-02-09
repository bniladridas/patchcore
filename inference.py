#!/usr/bin/env python3
"""Inference script to load saved Patchcore model."""

from pathlib import Path

import torch
import torchvision.transforms as T
from anomalib.models import Patchcore
from PIL import Image

CHECKPOINT_PATH = Path("models/patchcore_demo.pth")


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

    if "model_params" in checkpoint:
        params = checkpoint["model_params"]
    else:
        params = {
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.5,
        }

    model = Patchcore(
        backbone=params["backbone"],
        layers=params["layers"],
        coreset_sampling_ratio=params["coreset_sampling_ratio"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model, image_path):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        return output.pred_score.item()


if __name__ == "__main__":
    if not CHECKPOINT_PATH.exists():
        print("Model not found. Run train_model.py first.")
    else:
        model = load_model()
        print("Model loaded successfully!")

        img_path = "datasets/test/good/000.png"
        score = predict(model, img_path)
        print(f"Anomaly score for {img_path}: {score:.4f}")
