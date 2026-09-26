# Release checklist — v1.0.0

Each item names the evidence behind it. Most of them are re-checked automatically:

```bash
pytest                            # unit tests
ruff check src tests scripts app  # style
python scripts/audit_manifest.py  # leakage audit from the committed manifest
python scripts/check_release.py   # the claims on this page
```

## 1. Dataset provenance and licence

- [x] Source and version pinned — `configs/data.yaml` (`mohanty/PlantVillage`, revision `9e97599…`)
- [x] Checksums recorded and verified on download — `DATASET_NOTES.md`, `src/data/prepare.py`
- [x] Licence determined from a primary source — CC BY-SA 3.0, quoted in `LICENSING.md`
- [x] Release of derived model weights permitted — yes, under CC BY-SA 3.0 with attribution
- [x] Redistribution of split manifests permitted — yes, same licence; no images are redistributed
- [x] Attribution present — `CITATION.md`, `MODEL_CARD.md`, Hub model card
- [x] Pretrained-backbone caveat documented — `LICENSING.md` §4 (ImageNet terms)

## 2. Data integrity

- [x] Splits generated at the leaf level — `src/data/grouping.py`, `src/data/splits.py`
- [x] Leakage audit passes on the committed manifest — `data/splits/leakage_audit.json`
- [x] Audit re-run in CI — `.github/workflows/tests.yml` → `scripts/audit_manifest.py`
- [x] Residual leakage for images without leaf ids measured, not assumed — `data/splits/grouping_calibration.csv`
- [x] Test metrics computed only on the held-out split, and blocked if the audit fails — `src/evaluation/evaluate.py`
- [x] Every class present in every split — `data/splits/split_summary.json`

## 3. Models and results

- [x] Binary track trained and evaluated — `binary_mobilenet_v2`, `binary_efficientnet_b0`
- [x] Multiclass track trained and evaluated — `multiclass_mobilenet_v2`, `multiclass_efficientnet_b0`
- [x] Per-class precision/recall/F1 exported — `artifacts/reports/<run>/test/per_class_metrics.csv`
- [x] Macro and weighted averages, balanced accuracy, top-k, calibration — `metrics.json` per run
- [x] Confusion matrices exported for both tracks — `artifacts/figures/<run>/`
- [x] Robustness check with the background changed — `test_segmented`, `test_segmented_gray`
- [x] Experiment configs committed — `configs/`, and copied next to each report

## 4. Explainability and error analysis

- [x] Grad-CAM for correct and incorrect predictions — `artifacts/figures/<run>/gradcam_*.jpg`
- [x] Grad-CAM layer documented and justified — `src/models/factory.py`, `MODEL_CARD.md`
- [x] Quantitative attention measure, not just pictures — leaf-focus ratio in `failure_analysis.md`
- [x] Sample predictions gallery, not cherry-picked — `artifacts/sample_outputs/<run>/`
- [x] Failure analysis written per run — `artifacts/reports/<run>/failure_analysis.md`
- [x] Findings interpreted in prose — `docs/results.md`

## 5. Documentation

- [x] Model card complete, no placeholders, numbers matching the reports — `MODEL_CARD.md` (checked by `scripts/check_release.py`)
- [x] Limitations stated where the numbers are stated — `README.md`, `MODEL_CARD.md`, `docs/results.md`
- [x] Split policy documented — `data/splits/README.md`
- [x] Augmentation strategy documented, including what was rejected — `docs/augmentation.md`
- [x] Audit of the inherited state written down — `docs/previous_state_audit.md`
- [x] Notebooks run end to end on Colab, no absolute paths — `notebooks/`
- [x] No broken internal links — `scripts/check_release.py`

## 6. Repository hygiene

- [x] No dataset images committed — `.gitignore`, checked by `scripts/check_release.py`
- [x] No model weights committed (they go to the Hub) — same check
- [x] No credentials or personal data in the repository
- [x] Tests and lint green on `main` — CI
- [x] Planning issues closed with a status note — `.github/ISSUES/*.md`

## 7. Hugging Face release

- [x] Release folders built — `python scripts/build_hf_release.py` → `release/huggingface/`
- [x] Model card with Hub metadata, weights, ONNX, metrics and an inference example included
- [x] Demo Space prepared (Gradio app, example images with attribution)
- [ ] **Uploaded to the Hub** — `python scripts/upload_to_hub.py --model [--space REPO_ID]`, run by a maintainer after review
- [ ] **Hub link verified** by loading the model from a clean environment

## 8. Sign-off

- [ ] `python scripts/check_release.py` passes
- [ ] Repository reviewed by a maintainer
- [ ] Tag created:

```bash
git tag -a v1.0.0 -m "v1.0.0 — leaf-level leakage-free PlantVillage classifiers"
git push origin v1.0.0
```

The three unticked boxes are deliberate: publishing and tagging are decisions for the maintainers,
not steps in a pipeline.
