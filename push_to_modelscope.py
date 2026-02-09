#!/usr/bin/env python3
"""Push Patchcore model to ModelScope."""

from pathlib import Path
from modelscope.hub.api import HubApi

MODEL_PATH = Path("models/patchcore_demo.pth")
REPO_ID = "bniladridas/patch"


def push_model():
    api = HubApi()
    api.push_model(
        model_id=REPO_ID,
        model_dir=str(MODEL_PATH.parent),
        commit_message="feat: upload patchcore anomaly detection model",
    )
    print(f"Pushed to: https://modelscope.cn/models/{REPO_ID}")


if __name__ == "__main__":
    push_model()
