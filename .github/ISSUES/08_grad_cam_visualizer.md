---
title: "Build Grad-CAM visualizer for model explainability"
labels: ["task", "research"]
---

## Summary

Implement a Grad-CAM (Gradient-weighted Class Activation Mapping) visualizer in `src/visualization/` that generates heatmap overlays on leaf images to show which image regions the model focuses on when making predictions. Export visualizations to `artifacts/figures/`.

## Why It Matters

Grad-CAM provides interpretability for the model's predictions, helping verify that the model is using biologically meaningful leaf regions rather than background artifacts. It is a required explainability artifact for the model card and is particularly important given that PlantVillage images have controlled backgrounds that may introduce spurious cues.

## Acceptance Criteria

- [ ] Grad-CAM implementation in `src/visualization/` compatible with MobileNetV2 and EfficientNet B0
- [ ] Generates heatmap overlays on the original leaf image
- [ ] Supports both correct and incorrect prediction cases
- [ ] Exported visualizations include: original image, Grad-CAM heatmap, overlay blend
- [ ] A representative set of Grad-CAM samples exported to `artifacts/figures/`
- [ ] Visualizer is callable from a Colab notebook
- [ ] Documentation notes which convolutional layer is used for Grad-CAM extraction

## Dependencies

- Issue #03 or #04: Trained model checkpoint available
- Issue #01: Processed images available for visualization

## Notes

- Use `pytorch-grad-cam` library or a custom implementation
- Test on both correctly classified and misclassified examples
- Particularly examine cases where the model may be attending to background rather than leaf tissue
- Grad-CAM layer choice (last convolutional layer) should be documented and justified
