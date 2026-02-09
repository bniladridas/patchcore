#!/usr/bin/env python3
"""Simple anomaly detection demo using Patchcore."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from anomalib.models import Patchcore
from PIL import Image

DATA_DIR = Path("./datasets")
MODEL_DIR = Path("./models")


def create_dataset():
    """Create a simple image dataset."""
    (DATA_DIR / "train" / "good").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "test" / "good").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "test" / "defect").mkdir(parents=True, exist_ok=True)

    print("Creating dataset...")

    for i in range(20):
        img = Image.fromarray((np.random.rand(224, 224, 3) * 200 + 55).astype(np.uint8))
        img.save(DATA_DIR / "train" / "good" / f"{i:03d}.png")

    for i in range(10):
        img = Image.fromarray((np.random.rand(224, 224, 3) * 200 + 55).astype(np.uint8))
        img.save(DATA_DIR / "test" / "good" / f"{i:03d}.png")

    for i in range(10):
        arr = (np.random.rand(224, 224, 3) * 200 + 55).astype(np.uint8)
        arr[80:140, 80:140] = [255, 50, 50]
        img = Image.fromarray(arr)
        img.save(DATA_DIR / "test" / "defect" / f"{i:03d}.png")

    print(f"  Train/good: 20, Test/good: 10, Test/defect: 10")


class SimpleDataset:
    """Simple dataset that returns image tensor and label."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        for fname in sorted(self.data_dir.glob("*.png")):
            label = 0 if "good" in str(self.data_dir) else 1
            self.samples.append((str(fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    print("=" * 50)
    print("Anomaly Detection with Patchcore")
    print("=" * 50)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Creating dataset...")
    create_dataset()

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )

    print("\n[2/5] Loading Patchcore model...")
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.5,
    )
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n[3/5] Training (building memory bank)...")
    train_dataset = SimpleDataset(DATA_DIR / "train" / "good", transform)

    model.train()
    for idx in range(len(train_dataset)):
        img, _ = train_dataset[idx]
        img = img.unsqueeze(0)

        with torch.no_grad():
            features = model.model.feature_extractor(img)
            embedding = model.model.generate_embedding(features)
            reshaped = model.model.reshape_embedding(embedding)
            model.model.embedding_store.append(reshaped)

    model.model.subsample_embedding(sampling_ratio=0.5)
    print(f"  Coreset size: {model.model.memory_bank.shape[0]}")
    print("  Training complete!")

    print("\n[4/5] Running inference...")
    test_normal = SimpleDataset(DATA_DIR / "test" / "good", transform)
    test_anomaly = SimpleDataset(DATA_DIR / "test" / "defect", transform)

    normal_scores = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_normal)):
            img, _ = test_normal[idx]
            img = img.unsqueeze(0)
            output = model.model(img)
            normal_scores.append(output.pred_score.item())

        anomaly_scores = []
        for idx in range(len(test_anomaly)):
            img, _ = test_anomaly[idx]
            img = img.unsqueeze(0)
            output = model.model(img)
            anomaly_scores.append(output.pred_score.item())

    print("\n  Results:")
    print(
        f"    Normal images (n={len(normal_scores)}): Mean={np.mean(normal_scores):.4f}, Max={max(normal_scores):.4f}"
    )
    print(
        f"    Anomaly images (n={len(anomaly_scores)}): Mean={np.mean(anomaly_scores):.4f}, Min={min(anomaly_scores):.4f}"
    )

    threshold = 0.5
    detected = sum(1 for s in anomaly_scores if s > threshold)
    print(f"\n  Threshold: {threshold}")
    print(f"  Anomalies detected: {detected}/10")

    save_path = MODEL_DIR / "patchcore_demo.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "backbone": "resnet18",
                "layers": ["layer2", "layer3"],
                "coreset_sampling_ratio": 0.5,
            },
        },
        save_path,
    )
    print(f"\n  Model saved to: {save_path}")

    print("\n[5/5] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(normal_scores, bins=10, alpha=0.7, label="Normal", color="green")
    axes[0, 0].hist(anomaly_scores, bins=10, alpha=0.7, label="Anomaly", color="red")
    axes[0, 0].axvline(threshold, color="black", linestyle="--", label=f"Threshold")
    axes[0, 0].set_xlabel("Anomaly Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Score Distribution")
    axes[0, 0].legend()

    axes[0, 1].bar(
        ["Normal", "Anomaly"],
        [np.mean(normal_scores), np.mean(anomaly_scores)],
        color=["green", "red"],
        yerr=[np.std(normal_scores), np.std(anomaly_scores)],
    )
    axes[0, 1].set_ylabel("Mean Score")
    axes[0, 1].set_title("Mean Scores by Class")

    img = Image.open(DATA_DIR / "test" / "good" / "000.png")
    axes[1, 0].imshow(np.array(img))
    axes[1, 0].set_title(f"Normal Sample\nScore: {normal_scores[0]:.4f}")
    axes[1, 0].axis("off")

    img = Image.open(DATA_DIR / "test" / "defect" / "000.png")
    axes[1, 1].imshow(np.array(img))
    axes[1, 1].set_title(f"Anomaly Sample\nScore: {anomaly_scores[0]:.4f}")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("anomaly_results.png", dpi=150)
    print("  Saved to anomaly_results.png")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
