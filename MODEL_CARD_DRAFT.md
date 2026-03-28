# Model Card — Plant Disease Classification (Draft)

> ⚠️ **This model card is a draft placeholder.**
> It will be completed when a trained model is ready for release.
> Do not publish this card as-is.

---

## Model Overview

| Field | Value |
|---|---|
| **Model name** | Plant Disease Classifier (TBD) |
| **Version** | Draft — not yet trained |
| **Type** | Image classification (CNN fine-tuning) |
| **Framework** | PyTorch |
| **Base architecture** | MobileNetV2 or EfficientNet B0 (TBD after experiments) |
| **Training status** | Not yet trained |

---

## Intended Use

### Primary Use Case
- Classify plant leaf images as healthy or diseased (binary track)
- Identify specific disease categories from leaf images (multiclass track)

### Intended Users
- Researchers working on plant pathology and agricultural AI
- Developers building plant health monitoring tools
- Students and educators exploring transfer learning for image classification

### Out-of-Scope Uses
- Clinical or regulatory decision-making for crop management
- Real-time embedded or mobile deployment (not validated for this)
- Species outside the PlantVillage dataset distribution
- Field images without controlled conditions (generalization not yet validated)

---

## Dataset

| Field | Value |
|---|---|
| **Dataset** | PlantVillage |
| **Source** | [PlantVillage project](https://plantvillage.psu.edu/) — license TBD |
| **Image type** | Leaf images, controlled background |
| **Classes** | Multiple plant species and disease categories |
| **Split policy** | `leaf_id`-based train/val/test split to prevent leakage |

> ⚠️ Dataset license and redistribution terms must be verified before release. See `LICENSE_PLACEHOLDER.md`.

---

## Training Plan

_Placeholder — to be filled after training._

- **Optimizer:** TBD (planned: AdamW)
- **Learning rate:** TBD
- **Batch size:** TBD
- **Epochs:** TBD
- **Augmentation:** TBD (planned: horizontal flip, rotation, color jitter)
- **Hardware:** Google Colab (T4 or A100 GPU)
- **Regularization:** TBD

---

## Evaluation Plan

_Placeholder — to be filled after evaluation._

| Metric | Binary Track | Multiclass Track |
|---|---|---|
| Accuracy | TBD | TBD |
| Per-class Precision | TBD | TBD |
| Per-class Recall | TBD | TBD |
| Per-class F1 | TBD | TBD |
| Macro F1 | TBD | TBD |
| Confusion Matrix | TBD | TBD |

> ⚠️ No performance metrics are reported in this draft. Do not publish placeholder values.

---

## Limitations

- **Controlled-background bias:** The model is trained on lab-condition images. Performance on field images may be substantially lower.
- **Species coverage:** Limited to species and diseases present in PlantVillage.
- **Geographic generalization:** Not validated on images from different geographic regions or growing conditions.
- **Disease progression:** Does not account for disease progression stages or mixed infections.
- **Class imbalance:** PlantVillage has unequal class sizes; this will be addressed in training but may affect minority class performance.

---

## Ethical and Practical Caveats

- This model is a research prototype and is not validated for agricultural decision-making in production settings.
- Incorrect predictions may lead to incorrect disease treatment decisions if misused.
- The dataset's controlled conditions mean real-world generalization must be independently validated before deployment.
- Users should be aware that visual classification does not replace expert agronomic assessment.

---

## License

> ⚠️ **License is not yet determined.**
> See `LICENSE_PLACEHOLDER.md` for the current status and required verification steps.

---

## Contact

See the GitHub repository for issue reporting and contribution guidelines.
