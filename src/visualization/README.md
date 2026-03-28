# src/visualization/

This directory will contain all visualization utilities for model evaluation and explainability.

## Planned Contents

| Module | Purpose |
|---|---|
| `confusion_matrix.py` | Confusion matrix plot generation (binary and multiclass, normalized and raw counts) |
| `grad_cam.py` | Grad-CAM heatmap generation and image overlay |
| `gallery.py` | Sample predictions gallery layout and export |
| `utils.py` | Shared plotting utilities (figure sizing, color palettes, label rendering) |

## Output Locations

- Confusion matrix figures → `artifacts/figures/`
- Grad-CAM heatmaps → `artifacts/figures/`
- Sample predictions gallery → `artifacts/sample_outputs/`

## Design Notes

- All visualizers should be callable from Colab notebooks with a single function call.
- Output figures should be exported at sufficient resolution for reports and publications.
- No implementation code exists in this directory yet.

## Status

Empty placeholder. Implementation begins in Week 3 (see `ROADMAP.md`).
