"""Assemble the Hugging Face model repo and Space folders under release/.

    python scripts/build_hf_release.py

release/huggingface/model/   -> atilimai/plantvillage-leaf-disease-classifier
    README.md                   model card (copied from MODEL_CARD.md with Hub metadata)
    <run>/model.safetensors, config.json, model.onnx
    metrics/<run>.json          test metrics
    inference_example.py, requirements.txt
release/huggingface/space/   -> a Gradio Space
    app.py, requirements.txt, README.md, src/ (only what inference needs), examples/

Nothing is uploaded; see scripts/upload_to_hub.py for that.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import load_split  # noqa: E402

RELEASE = REPO / "release" / "huggingface"
RUNS = ["multiclass_mobilenet_v2", "binary_mobilenet_v2", "multiclass_efficientnet_b0", "binary_efficientnet_b0"]
SPACE_SOURCES = [
    "src/__init__.py", "src/utils/__init__.py", "src/utils/device.py",
    "src/data/__init__.py", "src/data/labels.py", "src/data/transforms.py",
    "src/models/__init__.py", "src/models/factory.py",
    "src/inference/__init__.py", "src/inference/predictor.py",
    "src/visualization/__init__.py", "src/visualization/grad_cam.py",
]

HUB_METADATA = """---
language: en
license: cc-by-sa-3.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - plant-disease
  - plantvillage
  - agriculture
  - image-classification
  - mobilenetv2
  - efficientnet
datasets:
  - mohanty/PlantVillage
metrics:
  - accuracy
  - f1
---

"""

SPACE_README = """---
title: PlantVillage Leaf Disease Classifier
emoji: 🍃
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-sa-3.0
---

Demo for [atilimai/plantvillage-leaf-disease-classifier](https://huggingface.co/atilimai/plantvillage-leaf-disease-classifier).
Source: https://github.com/atilimai/plant-ai-project
"""

INFERENCE_EXAMPLE = '''"""Minimal inference without the project code (PyTorch + torchvision only)."""

import json
import sys

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import models, transforms

REPO_ID = "atilimai/plantvillage-leaf-disease-classifier"
RUN = "multiclass_mobilenet_v2"  # or binary_mobilenet_v2, *_efficientnet_b0

config = json.load(open(hf_hub_download(REPO_ID, f"{RUN}/config.json")))
builder = {"mobilenet_v2": models.mobilenet_v2, "efficientnet_b0": models.efficientnet_b0}[config["arch"]]
model = builder(weights=None)
in_features = model.classifier[-1].in_features
model.classifier = torch.nn.Sequential(torch.nn.Dropout(config["dropout"]),
                                       torch.nn.Linear(in_features, len(config["class_names"])))
model.load_state_dict(load_file(hf_hub_download(REPO_ID, f"{RUN}/model.safetensors")))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(config["mean"], config["std"]),
])

image = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    probs = model(preprocess(image).unsqueeze(0)).softmax(dim=1)[0]
for p, i in zip(*probs.topk(3)):
    print(f"{config['class_names'][i]:55s} {p:.1%}")
'''


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_model_repo() -> None:
    out = RELEASE / "model"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    card = (REPO / "MODEL_CARD.md").read_text(encoding="utf-8")
    (out / "README.md").write_text(HUB_METADATA + card, encoding="utf-8")
    (out / "inference_example.py").write_text(INFERENCE_EXAMPLE, encoding="utf-8")
    (out / "requirements.txt").write_text("torch>=2.2\ntorchvision>=0.17\nsafetensors\nhuggingface_hub\npillow\n")

    for run in RUNS:
        exported = REPO / "models" / "exported" / run
        if not (exported / "config.json").exists():
            print(f"skip {run}: not exported")
            continue
        # model.onnx.data only exists if an exporter wrote the weights as a sidecar.
        for name in ("model.safetensors", "config.json", "model.onnx", "model.onnx.data"):
            if (exported / name).exists():
                copy(exported / name, out / run / name)
        metrics = REPO / "artifacts" / "reports" / run / "test" / "metrics.json"
        if metrics.exists():
            copy(metrics, out / "metrics" / f"{run}.json")
    for figure in ("multiclass_mobilenet_v2/confusion_matrix_test_normalized.png",
                   "multiclass_mobilenet_v2/gradcam_errors.jpg"):
        if (REPO / "artifacts" / "figures" / figure).exists():
            copy(REPO / "artifacts" / "figures" / figure, out / "figures" / figure.replace("/", "_"))
    print(f"model repo -> {out}")


def build_space() -> None:
    out = RELEASE / "space"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    app = (REPO / "app" / "app.py").read_text(encoding="utf-8")
    app = app.replace('MODEL_REPO = os.environ.get("PLANT_MODEL_REPO")',
                      'MODEL_REPO = os.environ.get("PLANT_MODEL_REPO", "atilimai/plantvillage-leaf-disease-classifier")')
    (out / "app.py").write_text(app, encoding="utf-8")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    (out / "requirements.txt").write_text(
        "torch>=2.2\ntorchvision>=0.17\nnumpy\npillow\nmatplotlib\nsafetensors\nhuggingface_hub\n")
    for rel in SPACE_SOURCES:
        copy(REPO / rel, out / rel)

    # One test image per crop as clickable examples. These are PlantVillage images
    # (CC BY-SA 3.0), redistributed with attribution in the Space README.
    manifest = load_split(REPO / "data" / "splits" / "manifest.csv", "test")
    manifest["crop"] = manifest["class_name"].str.split("___").str[0]
    picks = manifest.groupby("crop").sample(1, random_state=0).head(8)
    image_root = REPO / "data" / "raw" / "plantvillage"
    for _, row in picks.iterrows():
        if (image_root / row["path"]).exists():
            copy(image_root / row["path"], out / "examples" / f"{row['class_name']}.jpg")
    with open(out / "README.md", "a", encoding="utf-8") as f:
        f.write("\nExample images come from the PlantVillage test split "
                "(Hughes & Salathé, 2015; Mohanty et al., 2016), CC BY-SA 3.0.\n")
    print(f"space -> {out}")


def main():
    build_model_repo()
    build_space()
    summary = REPO / "artifacts" / "reports" / "summary.json"
    if summary.exists():
        df = pd.DataFrame(json.loads(summary.read_text(encoding="utf-8")))
        print(df[(df["split"] == "test") & (df["variant"] == "color")][["run", "accuracy", "macro_f1"]])


if __name__ == "__main__":
    main()
