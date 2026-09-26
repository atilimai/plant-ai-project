# Audit of the repository before the rework

Date: 2026-09-15. Baseline commit: `3d1f050` ("Add files via upload").

This records what was checked, what was wrong, and what was removed, so the git history makes
sense to anyone reading it later. Nothing here is about who did what; the problems are typical of
a project assembled from Colab notebooks under time pressure.

## Summary

| Area | State found | Consequence |
|---|---|---|
| Split files | `leaf_id` was a per-row counter (`inst_0`, `inst_1`, …) | The "leakage guard" grouped nothing |
| Split files | Colour, grayscale and segmented copies of the same photo were split independently | 87.4% of test rows had the same photo in train |
| Reported metrics | Multiclass model trained on a random image split, then scored on images from a different split | Test images overlapped the model's own training data; 95.8% accuracy is not a valid test result |
| Binary track | No binary model was trained; binary numbers were derived from the multiclass model | Issue #03 was not done |
| Figures | Confusion matrix, Grad-CAM and gallery PNGs were produced by `tests/test_pipeline.py` from random noise and random labels | Figures in `artifacts/` did not show any real model |
| Resolution | Images downscaled to 64×64 | Far below what ImageNet backbones expect; lesions barely visible |
| Documentation | README still said "scaffold only"; RELEASE_CHECKLIST ticked items that were not true (model card complete, leakage audit documented, no placeholders) | Status was not trustworthy |
| Licence | `DATASET_NOTES.md` said CC BY-NC-SA 3.0, `LICENSE_PLACEHOLDER.md` said CC BY-SA 3.0 | Contradiction; the paper and the dataset card say CC BY-SA 3.0 |

## Details

### 1. Split manifests (`data/splits/train_split.csv`, `test_split.csv`)

Produced by `notebooks/Boran_Data_Preparation.ipynb`. The notebook loaded the Hugging Face dataset,
looked for a `path` column (it is called `image_path`), did not find it, and fell back to

```python
df['leaf_id'] = [f"inst_{i}" for i in range(len(df))]
```

`GroupShuffleSplit` over unique ids is a plain random split, so the reported "intersection = 0"
check was true by construction. In addition the dataframe held all three PlantVillage variants
(colour, grayscale, segmented: 129,783 rows), so the same photo appeared up to three times.

Measured on the committed files:

* 19,753 of 22,106 distinct test photos also appear in train in another variant;
* 99.98% of test rows with an author leaf id have that leaf in train.

### 2. Baseline training (`notebooks/Ilgin_Baseline_Model.ipynb`)

* Split: `train_test_split(..., stratify=label)` on colour images, i.e. by image. On images with
  author leaf ids this puts 99.4% of test images next to a sibling from the same leaf in train.
* Input resolution 64×64, 3 epochs, no augmentation, Adam lr 1e-3 on the whole network.
* The checkpoint was saved twice, once as `binary_baseline_model.pth`, but the model was the
  38-class model.

### 3. Evaluation (`notebooks/Plant_Project_Evaluation.ipynb`)

* Loaded the baseline checkpoint above and scored it on `data/splits/test_split.csv`, a different
  split from the one the model was trained with. Most of those images were in the model's training
  set.
* File lookup used only `<class>/<file name>`, so the 8,595 grayscale test rows silently resolved
  to the colour photo with the same name (which the grayscale split had assigned independently);
  about 1,350 colour photos were scored twice. The glob only matched `*.JPG` and `*.png`, so
  lower-case `.jpg` files were dropped. The result was the 16,842 "test images" in the report.
* The binary report was derived from the multiclass predictions.

`artifacts/reports/evaluation_results.json` (accuracy 0.958) and
`binary_classification_report.txt` came from this notebook.

### 4. Figures

`tests/test_pipeline.py` generated random images (`np.random.rand(224, 224, 3)`), random labels
(`Saglikli`, `Pasli_Yaprak`, `Bakteriyel_Leke`) and random predictions, then saved the confusion
matrix, gallery and Grad-CAM figures into `artifacts/`. Those PNGs were committed as deliverables.

### 5. Source code

`src/evaluation` and `src/visualization` contained thin wrappers (hard-coded output paths, `plt.show()`
inside library functions, Grad-CAM calling `.numpy()` on CUDA tensors). There was no data, model,
training or inference code, and `tests/test_pipeline.py` imported a module that did not exist
(`src.utils`).

## What was removed

| Removed | Replaced by |
|---|---|
| `data/splits/train_split.csv`, `test_split.csv` | `data/splits/manifest.csv` + audit (`data/splits/README.md`) |
| `notebooks/Boran_Data_Preparation.ipynb`, `Ilgin_Baseline_Model.ipynb`, `Plant_Project_Evaluation.ipynb` | `notebooks/00`–`04`, backed by `src/` |
| All PNG/JSON/TXT under `artifacts/` | Outputs of `scripts/run_experiments.py` |
| `tests/test_pipeline.py` | `tests/` unit tests, run in CI |
| `src/evaluation/evaluator.py`, `reporter.py`, `src/visualization/utils.py` | `src/evaluation/evaluate.py`, `metrics.py`, `summarize.py`, `src/visualization/*` |

Everything removed is still available in git history at `3d1f050`.

## Also verified upstream

* `mohanty/PlantVillage` on Hugging Face now ships a `leaf_id` feature and an 80/20 split. It is a
  big improvement, but 243 of its 10,709 colour test images share a camera frame with train
  (the "copy" crops), and 25% of its test images have no leaf id, so for those it is effectively an
  image-level split. We therefore rebuild groups ourselves.
* Grayscale and segmented images are derived from the colour photos; mixing variants across splits
  leaks by definition. We train on colour only and use segmented copies only for a robustness check.
