---
title: "Build confusion matrix visualizer"
labels: ["task"]
---

## Summary

Implement a confusion matrix visualizer in `src/visualization/` that generates publication-quality confusion matrix plots for both the binary and multiclass classification tracks. Export plots to `artifacts/figures/`.

## Why It Matters

A confusion matrix reveals which specific class pairs are most frequently confused, enabling targeted diagnosis of model weaknesses. It is a standard and expected artifact in any classification project and is required for the model card.

## Acceptance Criteria

- [ ] Confusion matrix visualizer implemented in `src/visualization/`
- [ ] Supports both binary (2×2) and multiclass (N×N) configurations
- [ ] Confusion matrix is normalized (by row) and also shows raw counts (configurable)
- [ ] Class labels are readable on both axes
- [ ] Plots exported as PNG or PDF to `artifacts/figures/`
- [ ] Visualizer is callable from a Colab notebook with a single function call
- [ ] Example confusion matrices for both tracks committed to `artifacts/figures/`

## Dependencies

- Issue #06: Evaluation metrics pipeline must produce predictions and ground-truth labels
- Issue #03 or #04: At least one trained model checkpoint available for generating predictions

## Notes

- Use matplotlib or seaborn for plotting
- Consider including a title with model name and dataset split information
- For the multiclass track, ensure class label font size is legible when there are many classes
- Export at sufficient resolution for inclusion in reports and the model card
