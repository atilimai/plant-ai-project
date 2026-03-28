# Project Plan — Plant Disease Classification

## Goal

Develop a plant disease classification system using the PlantVillage dataset and transfer learning.
Deliver a reproducible pipeline covering data ingestion, model training, evaluation, visualization, and packaging, suitable for sharing via GitHub and Hugging Face.

---

## Non-Goals

- Real-time inference deployment or edge/mobile optimization
- New dataset collection or annotation
- Multi-modal or text-based disease explanation generation
- Production-grade API or web service
- Assigning tasks to specific individuals

---

## Definition of Done

The project is considered complete when all of the following are true:

- [ ] Dataset is downloaded, split by `leaf_id`, and split manifests are committed to `data/splits/`
- [ ] Binary classification baseline is trained and evaluated
- [ ] Multiclass classification baseline is trained and evaluated
- [ ] Per-class precision, recall, F1, and confusion matrix are computed and exported to `artifacts/`
- [ ] Grad-CAM visualizations are generated for representative samples
- [ ] Sample predictions gallery is documented in a notebook
- [ ] Failure case analysis is documented
- [ ] Model card is completed and reviewed
- [ ] Dataset license and attribution are verified
- [ ] No train/test leakage has been confirmed via `leaf_id` audit
- [ ] RELEASE_CHECKLIST.md is fully checked off
- [ ] Repository is clean and ready for public review

---

## Weekly Milestone Outline

### Week 1 — Foundation and Dataset Handling
- Set up repository structure (this scaffold)
- Download PlantVillage dataset and verify integrity
- Implement `leaf_id`-based train/val/test split logic
- Document split policy in `data/splits/README.md`
- Define data transforms and augmentation strategy
- Create `00_dataset_inspection.ipynb` content

### Week 2 — Baseline Training Setup
- Implement dataset loader in `src/data/`
- Define MobileNetV2 and EfficientNet B0 model wrappers in `src/models/`
- Implement training loop in `src/training/`
- Run binary classification baseline experiment
- Run multiclass classification baseline experiment
- Save checkpoints to `models/checkpoints/`
- Log experiment configs to `configs/`

### Week 3 — Evaluation and Visualizers
- Implement evaluation runner in `src/evaluation/`
- Compute per-class precision, recall, F1
- Generate confusion matrix and export to `artifacts/figures/`
- Implement Grad-CAM in `src/visualization/`
- Generate Grad-CAM samples for correct and incorrect predictions
- Document evaluation in `03_evaluation_plan.ipynb`

### Week 4 — Demo and Packaging
- Build sample predictions gallery in a notebook
- Document failure case analysis
- Draft model card in `MODEL_CARD_DRAFT.md`
- Begin Hugging Face packaging preparation
- Confirm dataset license and attribution

### Week 5 — Final Validation and Release Prep
- Complete `RELEASE_CHECKLIST.md` review
- Confirm no data leakage via `leaf_id` audit
- Finalize model card
- Finalize evaluation reports in `artifacts/reports/`
- Prepare Hugging Face release materials (pending license confirmation)
- Tag release on GitHub

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Data leakage via image-level splits | High | High | Enforce `leaf_id`-based splits from day one |
| Dataset license restricts public release | Medium | High | Verify license early in Week 1 |
| Controlled-background images inflate metrics | High | Medium | Document limitation clearly; consider field image testing |
| Colab session timeouts during training | Medium | Medium | Save checkpoints frequently; use Colab Pro if needed |
| Model underfits due to limited augmentation | Medium | Medium | Plan augmentation strategy in Week 1 |
| EfficientNet B0 vs MobileNetV2 comparison complexity | Low | Low | Run both, pick best for demo |

---

## Release Criteria

Before tagging a public release:

1. All items in `RELEASE_CHECKLIST.md` are checked
2. Dataset license and attribution confirmed
3. No data leakage confirmed by `leaf_id` audit
4. Model card is complete
5. Evaluation artifacts are exported and committed
6. Repository passes a final documentation review
