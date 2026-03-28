# Dataset Notes — PlantVillage

> These notes document what is known and what must be verified about the PlantVillage dataset before use and release.

---

## Dataset Source

| Field | Value |
|---|---|
| **Name** | PlantVillage |
| **Homepage** | [https://plantvillage.psu.edu/](https://plantvillage.psu.edu/) |
| **Paper** | Hughes & Salathé, 2015 — "An open access repository of images on plant health" |
| **Download source** | TBD — verify official access method (Kaggle, direct download, or Hugging Face Datasets) |
| **License** | ⚠️ TBD — must be verified before release |
| **Version used** | TBD — record exact version and hash |

---

## Classification Framing

PlantVillage supports two classification framings relevant to this project:

### Binary Framing — Healthy vs. Unhealthy
- All images are relabeled as `healthy` or `unhealthy`
- Simpler problem; useful as a first baseline and for fast triage use cases

### Multiclass Framing — Per-Disease Category
- Each image retains its original class label (e.g., `Tomato___Early_blight`, `Potato___Late_blight`)
- Harder problem; more practically informative

---

## ⚠️ leaf_id Split Integrity — Critical Constraint

The PlantVillage dataset contains **multiple images per physical leaf** (different angles, lighting, and crops of the same leaf).

**If splits are made at the image level without respecting leaf identity, the same physical leaf can appear in both the training set and the test set. This causes data leakage and leads to artificially inflated performance metrics.**

**Requirement:**
- All train/val/test splits must be made at the **`leaf_id` level**
- Split manifests must group all images from the same leaf into the same partition
- This must be enforced programmatically and documented in `data/splits/README.md`

See draft issue: `.github/ISSUES/02_leaf_id_leakage_guard.md`

---

## ⚠️ Controlled Background Limitation

All PlantVillage images are captured under **controlled laboratory conditions** with uniform, non-natural backgrounds.

This has several implications:
- A model trained exclusively on PlantVillage may learn to use background cues, not just leaf features
- Lab-measured accuracy may be substantially higher than field accuracy
- Performance claims made on the PlantVillage test set should be clearly qualified as lab-condition performance
- Real-world generalization should be independently evaluated before any deployment claim

---

## ⚠️ Real-World Generalization Warning

Published research consistently shows that models trained on PlantVillage perform significantly worse on:
- Field images taken with mobile phones
- Images with natural backgrounds
- Images under different lighting and weather conditions
- Images from different geographic regions or growing seasons

This project does not attempt to address real-world generalization in its current scope, but this limitation must be clearly stated in the model card and any published results.

---

## Dataset Statistics (To Be Filled)

| Stat | Value |
|---|---|
| Total images | TBD |
| Number of plant species | TBD |
| Number of disease classes | TBD |
| Number of healthy classes | TBD |
| Image resolution | TBD |
| Approximate class balance | TBD |

---

## Notes for Data Ingestion

- Record the exact download source URL and checksum
- Do not commit raw dataset images to the repository
- Commit only split manifests and metadata to `data/splits/`
- Document any preprocessing steps that alter the original data
