# src/visualization

Figure code. Every function takes data and returns a matplotlib figure, and only writes a file when
`save_path` is given — so the same call works in a notebook and in a script.

| Module | Figures |
|---|---|
| `confusion_matrix.py` | Row-normalised and raw-count matrices; for 38 classes only non-zero off-diagonal cells are annotated |
| `plots.py` | Training curves, per-class F1 bars, reliability diagram |
| `grad_cam.py` | Grad-CAM (own implementation, ~40 lines of hooks), overlay, leaf-focus measure, image/heatmap/overlay triplet |
| `gallery.py` | Confidence-stratified prediction gallery: confident hits, borderline hits, the most confident errors |

`leaf_focus()` uses the background-removed copy of an image as a leaf mask and reports how much of
the Grad-CAM mass falls on the leaf. That turns "is the model looking at the background?" into a
number; the answer for our models is in `docs/results.md`.
