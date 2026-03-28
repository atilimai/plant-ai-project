# data/splits/

This directory contains train/val/test split manifest files for the PlantVillage dataset.

## ⚠️ Split Integrity Constraint — leaf_id

**All splits in this directory must be generated at the `leaf_id` level, not the image level.**

The PlantVillage dataset contains multiple images per physical leaf (different angles, crops, and lighting).
Splitting by image without respecting `leaf_id` causes data leakage — the same physical leaf appears in both
the training set and the test set — leading to artificially inflated performance metrics.

**This is a critical research constraint. Any model evaluated without this guard produces invalid metrics.**

## Planned Contents

| File | Purpose |
|---|---|
| `train_manifest.csv` | List of image paths and labels assigned to the training split |
| `val_manifest.csv` | List of image paths and labels assigned to the validation split |
| `test_manifest.csv` | List of image paths and labels assigned to the test split |
| `split_policy.md` | Documentation of the split ratios, seed, and `leaf_id` grouping logic |
| `leakage_audit.txt` | Output of the leakage audit confirming no `leaf_id` overlap between splits |

## Split Ratios (Planned)

| Split | Approximate share |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Ratios are defined at the `leaf_id` level.

## Status

Empty placeholder. Split files will be generated in Week 1 after dataset ingestion (Issue #01, #02).
