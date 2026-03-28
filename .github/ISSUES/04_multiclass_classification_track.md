---
title: "Multiclass classification track: per-disease category"
labels: ["task", "research"]
---

## Summary

Implement, train, and evaluate a multiclass plant disease classifier that predicts the specific disease category for each leaf image. This is the primary classification track for detailed disease identification.

## Why It Matters

Fine-grained disease identification is more useful for practical agricultural applications than binary triage. It enables targeted treatment recommendations and more informative diagnostic outputs.

## Acceptance Criteria

- [ ] All disease class labels are mapped and enumerated from the dataset
- [ ] A fine-tuned MobileNetV2 or EfficientNet B0 model is trained on all multiclass categories
- [ ] Training and validation loss and accuracy are logged per epoch
- [ ] Best checkpoint is saved to `models/checkpoints/`
- [ ] Per-class precision, recall, and F1 are computed on the held-out test set
- [ ] Macro-averaged and weighted F1 are reported
- [ ] Confusion matrix is generated and exported to `artifacts/figures/`
- [ ] Results are documented in `artifacts/reports/`
- [ ] Experiment config is saved to `configs/`

## Dependencies

- Issue #01: Dataset ingestion complete
- Issue #02: Leakage guard implemented and splits are valid
- Issue #05: Augmentation strategy defined (recommended before training)

## Notes

- Class imbalance is expected in the multiclass setting — document the class distribution
- Consider weighted cross-entropy or focal loss to address imbalance
- Compare MobileNetV2 vs. EfficientNet B0 performance and document in the report
- Multiclass results are the primary performance benchmark for the model card
