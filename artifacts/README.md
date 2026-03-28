# artifacts/

This directory stores all generated artifacts from training, evaluation, and visualization.

## Subdirectories

### `figures/`
Saved plots and visualizations:
- Confusion matrices (binary and multiclass tracks)
- Grad-CAM heatmap overlays
- Training curves (loss and accuracy per epoch)
- Per-class metric bar charts

### `reports/`
Evaluation report outputs:
- Per-class precision, recall, F1 CSV/JSON exports
- Macro and weighted aggregate metrics
- Failure case analysis summary documents

### `sample_outputs/`
Sample prediction outputs:
- Sample predictions gallery images
- Representative correct and incorrect prediction grids
- Grad-CAM overlays for demo purposes

## Notes

- Artifact files are **not** tracked by git by default (add a `.gitignore` rule if needed)
- Commit final, reviewed artifacts when preparing for release
- Do not commit intermediate or experimental artifacts to the main branch

## Status

Empty placeholder. Artifacts will be generated progressively starting from Week 3 (see `ROADMAP.md`).
