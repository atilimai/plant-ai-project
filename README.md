# Plant Disease Classification

> **⚠️ This repository is currently a scaffold only.**
> No implementation code, trained models, or evaluation results exist yet.
> All notebooks, source folders, and model directories are placeholders.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Scope](#scope)
- [Dataset — PlantVillage](#dataset--plantvillage)
- [Planned Classification Tracks](#planned-classification-tracks)
- [Planned Visualizers](#planned-visualizers)
- [Planned Tasks](#planned-tasks)
- [Repository Structure](#repository-structure)
- [Colab-First Workflow](#colab-first-workflow)
- [⚠️ Data Leakage Warning — leaf_id Split Integrity](#️-data-leakage-warning--leaf_id-split-integrity)
- [⚠️ License and Attribution Warning](#️-license-and-attribution-warning)
- [Planned Deliverables](#planned-deliverables)
- [Contributing](#contributing)

---

## Project Overview

This project develops a plant disease classification system from leaf images using the PlantVillage dataset.
The model will be able to distinguish healthy leaves from diseased ones (binary track) and identify specific disease categories (multiclass track).

The planned modeling approach uses transfer learning with **MobileNetV2** or **EfficientNet B0** implemented in **PyTorch**.

---

## Scope

**In scope:**
- Binary classification: healthy vs. unhealthy leaves
- Multiclass classification: per-disease category prediction
- Transfer learning fine-tuning with pre-trained CNN backbones
- Per-class evaluation metrics (precision, recall, F1, confusion matrix)
- Explainability via Grad-CAM visualizations
- Failure case documentation
- Sample prediction galleries
- Colab-ready notebook workflow

**Out of scope:**
- Real-time inference deployment
- Mobile or edge device optimization
- Multi-modal or textual disease description generation
- New dataset collection or annotation

---

## Dataset — PlantVillage

**Why PlantVillage is useful:**
- Large-scale, publicly available benchmark dataset for plant disease research
- Covers dozens of plant species and disease categories
- Supports both binary (healthy/unhealthy) and multiclass (per-disease) framing
- Widely used in the literature, enabling comparability

**Why PlantVillage is limited:**
- Images are taken under controlled laboratory conditions with uniform backgrounds
- Real-world generalization may be significantly weaker than lab-measured accuracy
- Controlled backgrounds may inflate classification performance compared to field deployment
- Dataset may not capture disease progression stages or mixed infections
- Geographic and species diversity may be limited

See [DATASET_NOTES.md](DATASET_NOTES.md) for full notes.

---

## Planned Classification Tracks

### Binary Track — Healthy vs. Unhealthy
- Input: leaf image
- Output: `healthy` or `unhealthy`
- Use case: fast triage screening

### Multiclass Track — Disease Category
- Input: leaf image
- Output: specific disease class label (e.g., `Tomato_Early_Blight`, `Potato_Late_Blight`, etc.)
- Use case: precise disease identification

Both tracks will be developed and evaluated independently.

---

## Planned Visualizers

| Visualizer | Purpose |
|---|---|
| Confusion matrix | Per-class error analysis |
| Per-class precision / recall / F1 | Performance breakdown |
| Grad-CAM heatmaps | Model explainability on leaf images |
| Sample predictions gallery | Correct and incorrect prediction display |
| Failure case analysis | Systematic review of model errors |

---

## Planned Tasks

> Tasks are not assigned to any individual. See [PROJECT_PLAN.md](PROJECT_PLAN.md) and [ROADMAP.md](ROADMAP.md).

- [ ] Dataset ingestion and split policy implementation (respecting leaf_id)
- [ ] Leakage guard: enforce leaf_id-based train/val/test splits
- [ ] Binary classification baseline
- [ ] Multiclass classification baseline
- [ ] Augmentation strategy design and implementation
- [ ] Evaluation metrics pipeline (precision, recall, F1, confusion matrix)
- [ ] Confusion matrix visualizer
- [ ] Grad-CAM visualizer
- [ ] Sample predictions gallery notebook
- [ ] Failure case analysis report
- [ ] Colab notebook structure
- [ ] Model card drafting
- [ ] License and attribution review
- [ ] Hugging Face packaging preparation
- [ ] Final release checklist execution

---

## Repository Structure

```
plant-disease-classification/
├── .github/
│   ├── ISSUE_TEMPLATE/        # Issue templates (task, research, bug, docs, release)
│   └── ISSUES/                # Draft issues for planning
├── configs/                   # Hyperparameter and experiment config files
├── data/
│   ├── raw/                   # Original unmodified dataset files
│   ├── interim/               # Intermediate transformations
│   ├── processed/             # Final processed data ready for training
│   └── splits/                # Train / val / test split manifests
├── docs/                      # Extended documentation
├── notebooks/                 # Colab-ready experiment notebooks (placeholders)
├── src/
│   ├── data/                  # Dataset loading, transforms, split utilities
│   ├── models/                # Model architecture definitions
│   ├── training/              # Training loops and configurations
│   ├── evaluation/            # Metrics computation and evaluation runners
│   ├── visualization/         # Confusion matrix, Grad-CAM, galleries
│   └── inference/             # Prediction and export utilities
├── tests/                     # Unit and integration tests
├── artifacts/
│   ├── figures/               # Saved plots and visualizations
│   ├── reports/               # Evaluation report outputs
│   └── sample_outputs/        # Sample prediction outputs
├── app/                       # Demo app placeholder
├── models/
│   ├── checkpoints/           # Saved model checkpoints
│   └── exported/              # ONNX or TorchScript exports
└── references/                # Papers, dataset cards, related work
```

---

## Colab-First Workflow

This project is designed to run primarily in **Google Colab** notebooks.

- All experiment notebooks are located in `notebooks/` and will be developed as Colab-compatible `.ipynb` files
- Dataset loading assumes Colab Drive mounting or Hugging Face Datasets access
- No local GPU is assumed; all training plans target Colab T4/A100 runtimes
- Configs in `configs/` will use simple formats (YAML or JSON) readable without additional tooling

---

## ⚠️ Data Leakage Warning — leaf_id Split Integrity

> **This is a critical research constraint.**

The PlantVillage dataset contains multiple images per physical leaf (different angles, lighting, and crops).
**Splitting by image without respecting `leaf_id` causes data leakage**, where the same physical leaf appears in both training and test sets, leading to artificially inflated performance metrics.

**All train/val/test splits in this project must be made at the `leaf_id` level**, not the image level.
Split manifests in `data/splits/` must encode this constraint.

See draft issue: `.github/ISSUES/02_leaf_id_leakage_guard.md`

---

## ⚠️ License and Attribution Warning

> **Before any public release, the following must be verified:**

- [ ] Confirm the PlantVillage dataset license and terms of use
- [ ] Confirm attribution requirements for the dataset
- [ ] Confirm that model weights derived from PlantVillage comply with the license
- [ ] Confirm Hugging Face release is permitted under the dataset license
- [ ] Add proper attribution to all published artifacts

See [LICENSE_PLACEHOLDER.md](LICENSE_PLACEHOLDER.md) and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

---

## Planned Deliverables

### GitHub
- Trained model weights (if license permits)
- Evaluation reports with confusion matrices and per-class metrics
- Grad-CAM visualizations
- Sample prediction galleries
- Failure case analysis notebook
- Completed model card

### Hugging Face (pending license verification)
- Model hosted on Hugging Face Hub
- Dataset card (if redistribution is permitted)
- Demo Space with Gradio interface (optional)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.
