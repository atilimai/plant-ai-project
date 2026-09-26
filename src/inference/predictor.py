"""Load a trained model and classify leaf images.

A model can come from either
* a training checkpoint (``models/checkpoints/<run>/best.pt``), or
* an exported release folder with ``config.json`` + ``model.safetensors``
  (what we publish on the Hugging Face Hub, see ``src/inference/export.py``).

    predictor = Predictor.load("models/exported/multiclass_mobilenet_v2")
    predictor.predict("leaf.jpg", top_k=3)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.labels import pretty_name
from src.data.transforms import build_transforms
from src.models.factory import build_model, gradcam_layer
from src.utils.device import pick_device
from src.visualization.grad_cam import GradCAM, overlay


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Return (model in eval mode, meta dict) from a .pt checkpoint or an export folder."""
    path = Path(path)
    device = torch.device(device)
    if path.is_dir():
        from safetensors.torch import load_file

        with open(path / "config.json", encoding="utf-8") as f:
            meta = json.load(f)
        state_dict = load_file(path / "model.safetensors", device=str(device))
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        meta, state_dict = ckpt["meta"], ckpt["model_state"]

    model = build_model(meta["arch"], len(meta["class_names"]), pretrained=False, dropout=meta.get("dropout", 0.2))
    model.load_state_dict(state_dict)
    return model.to(device).eval(), meta


@dataclass
class Prediction:
    label: str
    display_name: str
    confidence: float
    top_k: list[tuple[str, float]]


class Predictor:
    def __init__(self, model: torch.nn.Module, meta: dict, device: torch.device):
        self.model, self.meta, self.device = model, meta, device
        self.class_names = meta["class_names"]
        self.transform = build_transforms(train=False, image_size=meta["image_size"])

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> Predictor:
        dev = pick_device(device)
        model, meta = load_checkpoint(path, dev)
        return cls(model, meta, dev)

    def _batch(self, images) -> tuple[torch.Tensor, list[np.ndarray]]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        pil = [Image.open(i).convert("RGB") if isinstance(i, (str, Path)) else i.convert("RGB") for i in images]
        size = self.meta["image_size"]
        rgb = [np.asarray(p.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0 for p in pil]
        return torch.stack([self.transform(p) for p in pil]).to(self.device), rgb

    @torch.no_grad()
    def predict_proba(self, images) -> np.ndarray:
        batch, _ = self._batch(images)
        return self.model(batch).softmax(dim=1).cpu().numpy()

    def predict(self, images, top_k: int = 3) -> list[Prediction]:
        probs = self.predict_proba(images)
        return [self._to_prediction(p, top_k) for p in probs]

    def explain(self, image, target: int | None = None, top_k: int = 5) -> tuple[Prediction, np.ndarray, np.ndarray]:
        """Prediction plus (Grad-CAM heatmap, overlay image) for a single image."""
        batch, rgb = self._batch(image)
        with GradCAM(self.model, gradcam_layer(self.model)) as cam:
            cams, probs = cam(batch, None if target is None else [target])
        return self._to_prediction(probs[0], top_k), cams[0], overlay(rgb[0], cams[0])

    def _to_prediction(self, p: np.ndarray, top_k: int) -> Prediction:
        order = np.argsort(-p)[: max(1, top_k)]
        label = self.class_names[order[0]]
        return Prediction(
            label=label,
            display_name=display(label),
            confidence=float(p[order[0]]),
            top_k=[(self.class_names[i], float(p[i])) for i in order],
        )


def display(label: str) -> str:
    return pretty_name(label) if "___" in label else label.capitalize()
