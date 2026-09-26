# src/data

Everything between "a zip on the internet" and "a DataLoader".

| Module | What it does |
|---|---|
| `prepare.py` | CLI: download at a pinned revision, verify checksums, extract, group, split, audit, write `data/splits/` |
| `labels.py` | The 38 class names (order fixed forever), binary mapping, display names |
| `grouping.py` | Rebuilds a leaf id per image from the author leaf map, camera file names and duplicates |
| `splits.py` | Leaf-group split, contiguous split for id-less camera sequences, frame-buffer purge, leakage audit |
| `calibration.py` | Simulates the sequence rule on images that do have leaf ids, to choose the frame buffer |
| `dataset.py` | `PlantVillageDataset` (colour or segmented variant) and the DataLoader factory |
| `transforms.py` | Train/eval pipelines; see `docs/augmentation.md` for the reasoning |

Rules that are enforced in code, not just documented:

* augmentation is applied to the training split only (`build_transforms(train=...)`);
* a group never spans two splits, and `audit_splits` re-checks this from the manifest alone;
* images are never written back to disk, so the files always match the published checksums.

Start here: `data/splits/README.md` explains the split policy and its audit.
