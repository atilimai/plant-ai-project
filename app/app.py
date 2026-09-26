"""Gradio demo: upload a leaf photo, get a diagnosis and a Grad-CAM heatmap.

Run from the repository root:

    python app/app.py                         # uses models/exported/*
    PLANT_MODEL_REPO=atilimai/plantvillage-leaf-disease-classifier python app/app.py

The same file is the entry point of the Hugging Face Space (see scripts/build_hf_release.py).
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path

import gradio as gr
import numpy as np

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):  # Space layout keeps src/ next to app.py
    if (candidate / "src").is_dir():
        sys.path.insert(0, str(candidate))
        break

from src.inference.predictor import Predictor, display  # noqa: E402

MODEL_REPO = os.environ.get("PLANT_MODEL_REPO")
LOCAL_EXPORTS = Path(os.environ.get("PLANT_MODEL_DIR", HERE.parent / "models" / "exported"))
MODELS = {
    "Disease (38 classes)": "multiclass_mobilenet_v2",
    "Healthy vs diseased": "binary_mobilenet_v2",
}

DISCLAIMER = """
**Research prototype.** Trained on PlantVillage: single leaves photographed on a
plain background in a lab. Field photos (several leaves, soil, sky, hands) are out
of distribution and predictions on them are unreliable. Do not use this for
treatment decisions.
"""


@cache
def get_predictor(run_name: str) -> Predictor:
    """Prefer the Hub when a repo is configured, fall back to a local export."""
    if MODEL_REPO:
        try:
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(MODEL_REPO, allow_patterns=[f"{run_name}/*"]))
            if (root / run_name / "config.json").exists():
                return Predictor.load(root / run_name, device="cpu")
            print(f"{MODEL_REPO} has no {run_name}/ yet, trying the local export")
        except Exception as error:  # offline, gated repo, network hiccup
            print(f"Could not fetch {run_name} from {MODEL_REPO} ({error}), trying the local export")
    if not (LOCAL_EXPORTS / run_name / "config.json").exists():
        raise FileNotFoundError(
            f"No weights for {run_name}: neither on the Hub nor in {LOCAL_EXPORTS}. "
            f"Run `python -m src.inference.export --checkpoint models/checkpoints/{run_name}/best.pt`."
        )
    return Predictor.load(LOCAL_EXPORTS / run_name, device="cpu")


def available_models() -> list[str]:
    if MODEL_REPO:
        return list(MODELS)
    return [name for name, run in MODELS.items() if (LOCAL_EXPORTS / run / "config.json").exists()]


def diagnose(image, model_name: str):
    if image is None:
        return None, None, "Upload a leaf image first."
    try:
        predictor = get_predictor(MODELS[model_name])
    except FileNotFoundError as error:
        return None, None, f"**Model not available.** {error}"
    pred, _, heatmap = predictor.explain(image.convert("RGB"), top_k=5)
    scores = {display(label): p for label, p in pred.top_k}

    note = f"### {pred.display_name}\nconfidence **{pred.confidence:.1%}**"
    if pred.confidence < 0.6:
        note += "\n\nLow confidence: the photo may be unlike the training data, or the symptoms ambiguous."
    return scores, (heatmap * 255).astype(np.uint8), note


def build_demo() -> gr.Blocks:
    choices = available_models()
    examples_dir = HERE / "examples"
    examples = sorted(str(p) for p in examples_dir.glob("*.jpg")) if examples_dir.is_dir() else []

    with gr.Blocks(title="PlantVillage leaf disease classifier") as demo:
        gr.Markdown("# Leaf disease classifier\nMobileNetV2 fine-tuned on PlantVillage with a leaf-level, leakage-free split.")
        gr.Markdown(DISCLAIMER)
        if not choices:
            gr.Markdown("No exported model found. Run `python -m src.inference.export --checkpoint ...` "
                        "or set `PLANT_MODEL_REPO`.")
            return demo
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Leaf photo")
                model = gr.Radio(choices, value=choices[0], label="Model")
                button = gr.Button("Diagnose", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=image)
            with gr.Column():
                summary = gr.Markdown()
                label = gr.Label(num_top_classes=5, label="Top predictions")
                cam = gr.Image(label="Grad-CAM (where the model looked)")
        button.click(diagnose, inputs=[image, model], outputs=[label, cam, summary])
    return demo


if __name__ == "__main__":
    build_demo().launch()
