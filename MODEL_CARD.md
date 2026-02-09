# Patchcore Anomaly Detection Model

## Model Details

- **Model Type**: Image Anomaly Detection (Patchcore)
- **Backbone**: ResNet18
- **Layers Used**: layer2, layer3
- **Coreset Sampling Ratio**: 0.5
- **Framework**: anomalib >= 1.0.0
- **Training Date**: Trained on synthetic dataset

## Intended Use

This model is designed for **anomaly detection** in images. It learns from normal (non-defective) images and can identify anomalous regions in new images.

### Primary Use Cases
- Industrial quality control
- Defect detection in manufacturing
- Surface inspection
- Medical imaging anomaly detection

## Training Data

- **Normal Images**: 20 synthetic images (224x224 RGB)
- **Test Normal**: 10 synthetic images
- **Test Anomaly**: 10 synthetic images with defect patches

## Model Performance

Based on inference on test set:
- Normal images: Mean anomaly score ~13.87
- Anomaly images: Higher scores expected (threshold configurable)

## Usage

```python
import torch
from anomalib.models import Patchcore
from pathlib import Path

# Load model
checkpoint = torch.load("models/patchcore_demo.pth")
model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"], coreset_sampling_ratio=0.5)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    output = model(img_tensor)
    print(f"Anomaly score: {output.pred_score.item()}")
```

## Limitations

- Trained on synthetic data; performance may vary on real-world images
- Threshold should be tuned for specific use case
- May not generalize to anomaly types significantly different from training data

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
anomalib>=1.0.0
matplotlib>=3.7.0
numpy>=1.24.0
Pillow
```

## References

- [Patchcore Paper](https://arxiv.org/abs/2111.09085)
- [Anomalib Library](https://github.com/openvinotoolkit/anomalib)
