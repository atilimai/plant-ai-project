# Dataset notes — PlantVillage

What we use, where it comes from, and what is wrong with it. Everything here was checked against
the files we downloaded, not copied from the dataset card.

## Source

| Field | Value |
|---|---|
| Name | PlantVillage (colour variant) |
| Paper | Mohanty, Hughes & Salathé (2016), *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science 7:1419 |
| Original release | Hughes & Salathé (2015), arXiv:1511.08060 |
| Download | `https://huggingface.co/datasets/mohanty/PlantVillage` (mirror of github.com/spMohanty/PlantVillage-Dataset) |
| Revision used | `9e97599868962bd0079b8db4b7f1efa9185fa1e7` (pinned in `configs/data.yaml`) |
| File | `data.zip`, 2,184,723,441 bytes |
| SHA-256 | `fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84` |
| Leaf grouping metadata | `leaf_grouping/leaf-map.json`, SHA-256 `3b4b253683c6911744959a3870a92509b1faa1ee34e9f8585e21d0fe6337a25b` |
| Licence | CC BY-SA 3.0 (see `LICENSING.md`) |

`python -m src.data.prepare` downloads from that pinned revision and verifies both checksums before
extracting, so a silent change upstream cannot slip into a run.

## What is in it

| Stat | Value |
|---|---|
| Colour images | 54,305 |
| Classes | 38 (14 crops, 26 disease labels, 12 `*___healthy` labels) |
| Healthy share | 27.8% |
| Largest / smallest class | Orange Haunglongbing 5,507 / Potato healthy 152 (36×) |
| Resolution | 256×256 for every image |
| Format | JPEG, RGB (one image is RGBA) |
| Other variants in the zip | `grayscale` and `segmented`, 54,305 / 54,306 images, derived from the same photos |

Per-class counts are in `data/splits/split_summary.json`.

### Details worth knowing

* **The three variants are the same photographs.** Colour, grayscale and segmented versions of one
  photo must never be spread across splits. We train on colour only; the segmented copies are used
  for one robustness check (background removed).
* **1,196 colour images have no segmented counterpart**, 1,192 of them the whole Corn common rust
  class (their file names have no UUID prefix). The background-removed evaluation therefore covers
  37 of 38 classes.
* **21 pairs of byte-identical files** exist (same photo committed twice under different UUIDs).
  They are grouped together before splitting.
* **Some photos were cut into several tiles**, named `… .JPG`, `… copy.jpg`, `… copy 2.jpg`.
  These are different crops of one frame, not different leaves.
* **Tomato late blight contains a time series**: `GHLB_PS Leaf 23.7 Day 13` is one leaf photographed
  on several days.
* Class names carry small errors from the original release (`Haunglongbing` for Huanglongbing,
  `Corn_(maize)___Common_rust_` with a trailing underscore). We keep them exactly as published so
  labels match other work.

## Multiple images per leaf — the constraint that shapes this project

PlantVillage contains several photos of every physical leaf. The dataset does not ship a leaf id
in the folder structure, but the authors published one (`leaf-map.json`) covering 41,111 of the
54,305 colour images; the remaining 13,194 have no leaf id at all.

Splitting by image puts photos of the same leaf on both sides: measured on the images that do have
a leaf id, a random image-level split leaves **99.8%** of test images with a sibling in train. Such a
test set measures how well the model recognises leaves it has already seen.

How we rebuild leaf groups, what we do for the images without ids, and the audit that proves the
result is clean: `data/splits/README.md`.

## Limits of the data

* **Lab conditions.** Every photo is a single detached leaf on a uniform background under even
  lighting. Models trained on it can lean on background and framing cues; published work
  consistently reports large drops on field photos.
* **No field validation here.** This project does not contain any field images, so no claim about
  field performance can be supported. The closest proxy we report is accuracy on the
  background-removed copies of the same photos (`docs/results.md`).
* **Class imbalance**, 36× between the largest and smallest class. We report macro-F1 and balanced
  accuracy next to accuracy, and use class-weighted loss.
* **One label per image.** Mixed infections, disease stages and severity are not represented.
* **Uneven coverage.** 14 crops, mostly tomato, apple, grape and corn; only some crops have both
  healthy and diseased classes, so a model can partly identify the crop instead of the disease.
* **Label noise is unquantified.** Labels come from the original collection; we did not re-annotate.

## Rules we follow

* Raw images are never committed; only manifests under `data/splits/`.
* Any preprocessing that changes pixels happens at load time (`src/data/transforms.py`), so the
  files on disk always match the published checksums.
* Every reported number comes from the audited test split and is reproducible with
  `python scripts/run_experiments.py`.
