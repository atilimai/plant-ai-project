# Roadmap — Plant Disease Classification

> This roadmap outlines the planned work across 5 weeks.
> Tasks are not assigned to individuals.
> All timelines are targets, not commitments.

---

## Week 1 — Foundation and Dataset Handling

**Goal:** Establish a solid foundation before any modeling work begins.

- [ ] Finalize repository structure and documentation
- [ ] Download PlantVillage dataset from official source
- [ ] Verify dataset integrity (file counts, class distributions, image formats)
- [ ] Implement `leaf_id`-based train/val/test split generation
- [ ] Commit split manifest files to `data/splits/`
- [ ] Document split policy and leakage prevention strategy
- [ ] Define and document planned data augmentation strategy
- [ ] Create `configs/base_config.yaml` placeholder
- [ ] Populate `00_dataset_inspection.ipynb` with dataset overview

**Exit criteria:** Split manifests are committed. Leakage guard is in place. Dataset notes are documented.

---

## Week 2 — Baseline Training Setup

**Goal:** Run first training experiments for both classification tracks.

- [ ] Implement `PlantVillageDataset` class in `src/data/`
- [ ] Implement transforms and augmentation pipeline
- [ ] Define MobileNetV2 fine-tuning wrapper in `src/models/`
- [ ] Define EfficientNet B0 fine-tuning wrapper in `src/models/`
- [ ] Implement training loop with logging in `src/training/`
- [ ] Run binary classification (healthy vs. unhealthy) baseline
- [ ] Run multiclass classification baseline
- [ ] Save initial checkpoints to `models/checkpoints/`
- [ ] Log experiment hyperparameters to `configs/`

**Exit criteria:** Both classification tracks produce first checkpoint outputs.

---

## Week 3 — Evaluation and Visualizers

**Goal:** Measure performance rigorously and produce explainability artifacts.

- [ ] Implement evaluation runner in `src/evaluation/`
- [ ] Compute per-class precision, recall, F1 for both tracks
- [ ] Generate and export confusion matrices to `artifacts/figures/`
- [ ] Implement Grad-CAM in `src/visualization/`
- [ ] Generate Grad-CAM heatmaps for a representative sample of correct and incorrect predictions
- [ ] Export Grad-CAM figures to `artifacts/figures/`
- [ ] Populate `03_evaluation_plan.ipynb` with evaluation walkthrough

**Exit criteria:** Evaluation metrics and confusion matrices are exported. Grad-CAM visualizations are committed to `artifacts/`.

---

## Week 4 — Demo and Packaging

**Goal:** Prepare project for sharing and begin release documentation.

- [ ] Build sample predictions gallery in a Colab notebook
- [ ] Document failure cases (systematic misclassification patterns)
- [ ] Export sample outputs to `artifacts/sample_outputs/`
- [ ] Draft complete model card in `MODEL_CARD_DRAFT.md`
- [ ] Verify dataset license and attribution
- [ ] Begin Hugging Face packaging preparation
- [ ] Export model to ONNX or TorchScript in `models/exported/` (optional)
- [ ] Populate `04_demo_plan.ipynb`

**Exit criteria:** Model card draft is complete. License is verified. Demo artifacts are committed.

---

## Week 5 — Final Validation and Release Prep

**Goal:** Close all open items and prepare for public release.

- [ ] Complete all items in `RELEASE_CHECKLIST.md`
- [ ] Run final `leaf_id` leakage audit and document results
- [ ] Finalize evaluation reports in `artifacts/reports/`
- [ ] Finalize model card
- [ ] Finalize `CITATION.md`
- [ ] Prepare Hugging Face release materials (pending license confirmation)
- [ ] Tag GitHub release
- [ ] Close all planning issues with completion notes

**Exit criteria:** `RELEASE_CHECKLIST.md` fully checked. Repository is tagged and ready for public review.

---

## Future Directions (Post-Release)

These are not in scope for the current release but may be explored later:

- Field image testing to assess real-world generalization
- Multi-label classification for mixed infections
- Lightweight model distillation for mobile deployment
- Integration with a Hugging Face Space for interactive demos
- Cross-dataset generalization experiments
