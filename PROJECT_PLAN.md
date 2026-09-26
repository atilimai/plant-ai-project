# Project plan

## Goal

A plant disease classifier trained on PlantVillage whose reported numbers can be trusted: a
train/val/test split made at the level of the physical leaf, an audit that proves no leaf crosses
splits, per-class metrics, explainability artifacts, and an honest account of what the model cannot
do.

The accuracy number is not the deliverable. A reproducible pipeline and a defensible evaluation are.

## Non-goals

* Field deployment, mobile or edge optimisation.
* New data collection or re-annotation.
* Beating published PlantVillage leaderboards. Those numbers come from image-level splits and are
  not comparable with ours (see `CITATION.md`).
* A production API or service.

## Definition of done

| Item | Status |
|---|---|
| Dataset downloaded at a pinned revision, checksums verified | Done — `configs/data.yaml`, `src/data/prepare.py` |
| Leaf groups rebuilt; split made at leaf level; manifests committed | Done — `data/splits/manifest.csv` |
| Leakage audit implemented, passing, and re-run in CI | Done — `data/splits/leakage_audit.json`, `scripts/audit_manifest.py` |
| Binary track trained and evaluated | Done — `binary_mobilenet_v2`, `binary_efficientnet_b0` |
| Multiclass track trained and evaluated | Done — `multiclass_mobilenet_v2`, `multiclass_efficientnet_b0` |
| Per-class precision/recall/F1, macro and weighted averages, confusion matrices exported | Done — `artifacts/reports/`, `artifacts/figures/` |
| Grad-CAM for correct and incorrect predictions | Done — `artifacts/figures/<run>/gradcam_*.jpg` |
| Sample predictions gallery | Done — `artifacts/sample_outputs/<run>/` |
| Failure case analysis | Done — `artifacts/reports/<run>/failure_analysis.md`, `docs/results.md` |
| Model card completed with real numbers | Done — `MODEL_CARD.md` |
| Dataset licence verified from a primary source | Done — `LICENSING.md` |
| Unit tests and CI | Done — `tests/`, `.github/workflows/tests.yml` |
| Hugging Face release prepared | Done — `scripts/build_hf_release.py`; publishing is a manual step |
| Release checklist signed off | See `RELEASE_CHECKLIST.md` |

## How the work was organised

1. **Audit first.** Before writing any new code, the inherited repository was checked end to end.
   The split was leaky, the reported metrics were invalid and the committed figures were generated
   from random noise: `docs/previous_state_audit.md`.
2. **Data integrity.** Leaf grouping rebuilt from the authors' leaf map plus file-name evidence;
   images without a leaf id split by contiguous camera segments with a frame buffer, with the
   residual leakage measured rather than assumed: `data/splits/README.md`.
3. **Pipeline.** Dataset, transforms, model factory, trainer (resumable), evaluation, failure
   analysis, export, demo — all driven by YAML configs and callable from Colab.
4. **Experiments.** Two architectures × two tasks, identical recipe, one command:
   `python scripts/run_experiments.py`.
5. **Reporting.** Metrics, figures and the model card generated from the run outputs, with the
   limitations stated next to the numbers.

## Risks and how they were handled

| Risk | Outcome |
|---|---|
| Data leakage via image-level splits | Addressed: leaf-level split, audit in CI, residual leakage for id-less images measured at ≈0 |
| Dataset licence blocks release | Resolved: CC BY-SA 3.0, weights released under the same licence (`LICENSING.md`) |
| Lab backgrounds inflate accuracy | Not solvable with this dataset; quantified instead (background-removed evaluation, Grad-CAM leaf focus) and stated in the model card |
| Colab session timeouts | Training resumes from `last.pt`; checkpoints are written atomically |
| Class imbalance (36×) | Class-weighted loss; macro-F1 and balanced accuracy reported alongside accuracy |
| Small classes after the frame-buffer purge | Documented per class in `data/splits/README.md`; their metrics carry wider error bars |

## Release criteria

1. `RELEASE_CHECKLIST.md` fully checked.
2. Leakage audit passing on the committed manifest.
3. Model card free of placeholders, with numbers matching `artifacts/reports/summary.md`.
4. Licence and attribution verified (`LICENSING.md`).
5. CI green on `main`.
