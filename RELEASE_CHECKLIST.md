# Release Checklist — Plant Disease Classification

> Complete every item below before tagging a public release.
> Do not publish any artifact until this checklist is fully checked off.

---

## 1. Dataset Provenance and License

- [x] Confirm the official source and version of the PlantVillage dataset used
- [x] Record the download URL and file checksum in `DATASET_NOTES.md`
- [x] Verify the dataset license and terms of use
- [] Confirm that public release of model weights derived from this dataset is permitted
- [] Confirm that redistribution of preprocessed dataset splits is permitted (or exclude them from release)
- [x] Add required attribution text to `CITATION.md` and `MODEL_CARD_DRAFT.md`

---

## 2. License and Attribution

- [x] Assign a code license and add a `LICENSE` file to the repository
- [x] Confirm code license is compatible with the dataset license
- [] Assign a model license for released weights (if applicable)
- [] Confirm Hugging Face Hub hosting is permitted under the dataset license
- [x] Confirm all third-party libraries used have compatible licenses

---

## 3. Data Leakage Verification

- [x] Confirm all train/val/test splits are generated at the `leaf_id` level
- [] Run a leakage audit: verify no `leaf_id` appears in more than one partition
- [x] Document leakage audit results in `data/splits/README.md`
- [] Confirm evaluation metrics are computed only on the held-out test set

---

## 4. Evaluation Artifacts

- [ ] Final confusion matrices exported to `artifacts/figures/` for both classification tracks
- [ ] Per-class precision, recall, and F1 scores exported to `artifacts/reports/`
- [ ] Macro and weighted F1 scores documented
- [ ] Evaluation computed on the proper, leakage-free test set (confirmed above)

---

## 5. Sample Predictions and Failure Cases

- [ ] Sample predictions gallery committed to `artifacts/sample_outputs/` or a notebook
- [ ] Failure case analysis documented in a notebook or report
- [ ] Grad-CAM visualizations for representative samples exported to `artifacts/figures/`

---

## 6. Model Card

- [x] `MODEL_CARD_DRAFT.md` is complete with actual training and evaluation details
- [x] Intended use, limitations, and caveats are clearly stated
- [x] No placeholder text remains in the model card
- [x] Model card reviewed for accuracy

---

## 7. Hugging Face Release (if applicable)

- [ ] Dataset license permits Hugging Face hosting (confirmed above)
- [ ] Model uploaded to Hugging Face Hub with complete model card
- [ ] README and usage instructions added to the Hugging Face repository
- [ ] Demo Space set up (if applicable)

---

## 8. Repository Cleanliness

- [x] No raw dataset images committed to the repository
- [x] No API keys, credentials, or personal data in the repository
- [x] No broken links in documentation
- [x] All placeholder text in documentation has been replaced with real content
- [x] `CITATION.md` is finalized
- [ ] All open planning issues are closed or documented
- [ ] Repository is tagged with a release version

---

## Sign-Off

Once all items above are checked, tag the release:

```bash
git tag -a v1.0.0 -m "Release v1.0.0 — Plant Disease Classification"
git push origin v1.0.0
```
