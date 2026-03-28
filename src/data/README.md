# src/data/

This directory will contain all code related to dataset loading, preprocessing, and split management.

## Planned Contents

| Module | Purpose |
|---|---|
| `dataset.py` | `PlantVillageDataset` PyTorch Dataset class |
| `transforms.py` | Data augmentation and preprocessing pipelines |
| `splits.py` | `leaf_id`-based train/val/test split generation and validation |
| `loaders.py` | DataLoader factory functions |

## Key Constraints

- All train/val/test splits must be generated at the **`leaf_id` level** to prevent data leakage.
  See `DATASET_NOTES.md` and issue `#02_leaf_id_leakage_guard.md` for details.
- Augmentation must be applied **only to the training split**, never to validation or test splits.
- Dataset files must **not** be committed to the repository.

## Status

Empty placeholder. Implementation begins in Week 2 (see `ROADMAP.md`).
