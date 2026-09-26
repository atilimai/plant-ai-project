# notebooks

Five notebooks, meant to be read in order. They are thin: the logic lives in `src/`, the notebooks
call it and show the results, so a notebook cannot drift away from what the scripts do.

| Notebook | Covers | Issues |
|---|---|---|
| `00_dataset_inspection.ipynb` | Download and checksum, class distribution, sample images, leaf grouping, the committed split and its audit, augmentation check | #01, #02, #05 |
| `01_binary_experiment_plan.ipynb` | Healthy vs diseased: config, training, learning curves | #03 |
| `02_multiclass_experiment_plan.ipynb` | 38 classes: config, training, learning curves | #04 |
| `03_evaluation_plan.ipynb` | Test metrics, confusion matrices, calibration, Grad-CAM, failure analysis | #06–#08, #10 |
| `04_demo_plan.ipynb` | Inference from an exported model or the Hub, Grad-CAM, prediction gallery | #09, #14 |

## Running them

**Colab** — open with the badge at the top of each notebook. The first cell clones the repository
and installs `requirements.txt`. Pick a GPU runtime for 01–03 (*Runtime → Change runtime type*).
Notebook 00 downloads 2.2 GB of images.

**Locally** — from the repository root:

```bash
pip install -r requirements-dev.txt
jupyter lab notebooks/
```

The first cell walks up to the repository root, so notebooks work from any working directory and
contain no absolute paths.

## Conventions

* Training and evaluation cells are behind a `RUN_TRAINING` / `RUN_EVALUATION` flag, off by default,
  so opening a notebook never starts a multi-hour job by accident. The results shown come from the
  committed artifacts.
* Figures are embedded as downscaled JPEGs to keep the files reviewable in git.
* Test-set numbers only appear in notebook 03, and only after the leakage audit has passed.
