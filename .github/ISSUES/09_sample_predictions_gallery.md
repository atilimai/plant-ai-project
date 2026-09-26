---
title: "Create sample predictions gallery"
labels: ["task"]
---

## Summary

Produce a sample predictions gallery that displays a curated set of model predictions alongside ground-truth labels and confidence scores. The gallery should include both correct and incorrect predictions, organized in a readable notebook format and exported as figures to `artifacts/sample_outputs/`.

## Why It Matters

A predictions gallery provides human-interpretable evidence of model behavior that goes beyond aggregate metrics. It is a standard deliverable for ML projects and is required for the model card and Hugging Face release materials.

## Acceptance Criteria

- [x] Gallery shows at least 20 sample images with predicted label, true label, and confidence score
- [x] Includes a mix of correct predictions, borderline correct predictions, and clear errors
- [x] Includes samples from multiple plant species and disease classes
- [x] Exported as a figure grid to `artifacts/sample_outputs/`
- [x] Presented in a Colab-compatible notebook (linked from `notebooks/`)
- [x] Images are labeled clearly and legibly in the figure

## Dependencies

- Issue #03 or #04: Trained model checkpoint available
- Issue #01: Processed dataset available for sampling

## Notes

- Select samples manually or by confidence-based stratified sampling
- Avoid cherry-picking only easy or impressive examples — include representative failure modes
- Caption each image with the actual predicted class, ground-truth class, and confidence
- This gallery will be referenced in the model card and Hugging Face Space (if created)

---

## Status

Done. Confidence-stratified gallery (confident hits, borderline hits, most confident errors) in `artifacts/sample_outputs/<run>/prediction_gallery.jpg`, built by `src/visualization/gallery.py` and shown in `notebooks/04_demo_plan.ipynb`.
