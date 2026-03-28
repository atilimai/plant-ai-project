---
title: "Binary classification track: healthy vs. unhealthy"
labels: ["task", "research"]
---

## Summary

Implement, train, and evaluate a binary plant disease classifier that predicts whether a leaf image is healthy or unhealthy. This is the simpler of the two classification tracks and serves as a fast-iteration baseline.

## Why It Matters

A binary classifier is a practical triage tool that can flag potentially diseased leaves without needing fine-grained disease identification. It also provides a simpler baseline to validate the training pipeline before scaling to multiclass.

## Acceptance Criteria

- [ ] Binary label mapping (healthy / unhealthy) is implemented in the dataset loader
- [ ] A fine-tuned MobileNetV2 or EfficientNet B0 model is trained on the binary track
- [ ] Training and validation loss and accuracy are logged per epoch
- [ ] Best checkpoint is saved to `models/checkpoints/`
- [ ] Per-class precision, recall, and F1 are computed on the held-out test set
- [ ] Confusion matrix is generated and exported to `artifacts/figures/`
- [ ] Results are documented in `artifacts/reports/`
- [ ] Experiment config is saved to `configs/`

## Dependencies

- Issue #01: Dataset ingestion complete
- Issue #02: Leakage guard implemented and splits are valid

## Notes

- Healthy class: all images labeled as `healthy` in PlantVillage
- Unhealthy class: all images labeled with any disease category
- Check class balance and consider weighted loss or oversampling if imbalanced
- Document any class rebalancing strategy applied
