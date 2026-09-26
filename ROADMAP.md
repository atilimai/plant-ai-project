# Roadmap

## Where the project stands

The v1.0 scope is complete: a leaf-level, audited split; two tasks (healthy vs diseased, 38 disease
classes) trained with two backbones (MobileNetV2, EfficientNet-B0); per-class metrics, confusion
matrices, calibration, Grad-CAM and failure analysis on the held-out test set; exported weights,
a demo app and a Hugging Face release package.

Results: `artifacts/reports/summary.md` and `docs/results.md`. Numbers, caveats and intended use:
`MODEL_CARD.md`.

| Area | Delivered |
|---|---|
| Data | Pinned download with checksum verification, leaf grouping, leaf-level split, leakage audit in CI |
| Training | Config-driven, resumable, AMP, class weighting, early stopping on macro-F1 |
| Evaluation | Per-class metrics, macro/weighted averages, top-k, calibration (ECE, reliability), confusion matrices |
| Explainability | Grad-CAM on correct and incorrect predictions, Grad-CAM leaf-focus measurement against segmented masks |
| Robustness | Same test images with the background removed |
| Packaging | safetensors + ONNX export, Gradio demo, Hugging Face model repo and Space, model card |
| Quality | 36 unit tests, ruff, CI, manifest audit script |

## What we would do next

Ordered by how much they would change the conclusions, not by how easy they are.

### 1. Field images (the one that matters)

Every number in this repository describes single leaves on a plain background in a lab. That is a
ceiling on what can be claimed, and no amount of tuning on PlantVillage moves it. The next real step
is a few hundred annotated field photos, used purely as an external test set — not for training.
Until that exists, the honest statement stays "unknown field performance".

### 2. Background robustness

The background-removed evaluation tells us how much of the decision depends on the leaf itself.
Depending on how large that gap is, candidate interventions are: training on segmented images,
mixing colour and segmented copies, or background randomisation. Each needs the same leaf-level
audit, because the segmented copies share leaves with the colour ones.

### 3. Better use of the held-out data

* Cross-validation over the leaf groups instead of a single split, to put error bars on per-class
  F1 — some classes have fewer than 40 test images.
* Test-time augmentation and temperature scaling; calibration matters more than accuracy if the
  model is ever used with a confidence threshold.

### 4. Modelling

* Stronger backbones (ConvNeXt-T, EfficientNetV2-S) as a reference point for how much the current
  numbers are limited by capacity rather than by data.
* Hierarchical evaluation: crop first, then disease. Crop identity is easy; the disease decision
  inside a crop is the interesting part, and reporting it separately would be more informative.
* Distillation to a small model for mobile use, once there is field data worth deploying against.

### 5. Data quality

* The 13k images with no author leaf id are the weakest part of the split. A manual pass over a
  sample would tell us how good the camera-sequence heuristic really is.
* Label noise in PlantVillage is unquantified; a review of the most confident errors would show
  whether some of them are annotation mistakes rather than model failures.

## Out of scope

Field deployment, treatment recommendations, multi-label or severity prediction, and any claim about
performance outside the PlantVillage distribution.
