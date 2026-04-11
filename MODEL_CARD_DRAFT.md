# Model Card — Plant Disease Classification (Draft)

> ⚠️ **This model card is a draft placeholder.**
> It will be completed when a trained model is ready for release.
> Do not publish this card as-is.

---

## Model Overview

| Field | Value |
|---|---|
| **Model name** | Plant Disease Classifier |
| **Version** | v0.1 (Draft) |
| **Type** | Image classification (CNN fine-tuning) |
| **Framework** | PyTorch |
| **Base architecture** | MobileNetV2 or EfficientNet B0 (under experiments) |
| **Training status** | Not yet trained (pending experiments)|

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

_This section describes the planned training configuration. Final values will be updated after experiments._

- **Optimizer:** AdamW (planned)
- **Learning rate:** To be determined during hyperparameter tuning
- **Batch size:** To be determined based on GPU memory constraints
- **Epochs:** To be determined (early stopping may be applied)
- **Augmentation:** Horizontal flip, rotation, color jitter (planned)
- **Hardware:** Google Colab (T4 or A100 GPU)
- **Regularization:** To be determined (e.g., dropout, weight decay)

---

## Evaluation Plan

_Placeholder — to be filled after evaluation._
_This section outlines the evaluation strategy. Final metrics will be reported after model training and validation are completed._

| Metric | Binary Track | Multiclass Track |
|---|---|---|
| Accuracy | To be computed | To be computed |
| Per-class Precision | To be computed | To be computed |
| Per-class Recall | To be computed | To be computed |
| Per-class F1 | To be computed | To be computed |
| Macro F1 | To be computed | To be computed |
| Confusion Matrix | To be generated | To be generated |

**Evaluation Notes:**
- Both binary (disease vs. healthy) and multiclass (per-disease category) performance will be reported.
- Metrics will be computed on a held-out validation/test set.
- Class imbalance will be considered when interpreting results (macro vs weighted metrics).

> ⚠️ No performance metrics are reported in this draft. Do not publish placeholder values.

---

## Limitations

- **Controlled-background bias:** The model is primarily trained on lab-condition images (e.g., PlantVillage dataset). Performance on real-world field images with complex backgrounds, varying lighting conditions, and occlusions may be significantly lower.
- **Limited species coverage:** The model can only recognize plant species and diseases included in the training dataset. It will not generalize to unseen species or rare diseases.
- **Geographic generalization:** The model has not been validated on data from diverse geographic regions, climates, or agricultural practices, which may affect its robustness in real-world deployment.
- **Disease progression and co-infection:** The model does not explicitly account for different disease stages or multiple simultaneous infections, which may lead to incorrect predictions in such cases.
- **Class imbalance:** The dataset contains class imbalance, which may negatively impact performance on underrepresented classes despite mitigation strategies during training.
- **Image quality sensitivity:** Performance may degrade on low-resolution, blurry, or noisy images.
- **Domain shift risk:** Differences between training data distribution and real-world usage (domain shift) may reduce model reliability.

---

## Ethical and Practical Caveats

- **Research-only use:** This model is a research prototype and must not be used for real-world agricultural decision-making without further validation.
- **Risk of misdiagnosis:** Incorrect predictions may lead to inappropriate or harmful treatment decisions if the model is used without expert supervision.
- **Not a substitute for expertise:** The model does not replace professional agronomic or plant pathology expertise. Predictions should always be reviewed by qualified experts.
- **Generalization limitations:** Due to the controlled nature of the training dataset, the model’s real-world performance is uncertain and must be independently validated before deployment.
- **Bias and fairness considerations:** The model may underperform on underrepresented classes or conditions, potentially leading to unequal reliability across plant species or diseases.
- **Responsible use requirement:** Users are responsible for ensuring that the model is applied only in appropriate, non-critical contexts and with proper validation.
- **Transparency:** Model predictions should be accompanied by confidence scores and, where possible, explainability tools (e.g., Grad-CAM) to support interpretation.

---

## License

_The final license for this model will be determined after verifying the dataset and dependency licenses._

- The dataset (e.g., PlantVillage) license must be reviewed to ensure compliance with redistribution and commercial use.
- Any third-party components (pretrained models, libraries) must be checked for license compatibility.
- The final model will be released under an appropriate open-source license (e.g., MIT, Apache 2.0) once verification is complete.
- > See `LICENSE_PLACEHOLDER.md` for the current status and required verification steps.
> ⚠️ This model must not be publicly released until all licensing requirements are verified and documented.

---

## Contact

See the GitHub repository for issue reporting and contribution guidelines.
