# Licensing and attribution

Result of the licence review (issue #13). It covers four things that have different licences: the
code in this repository, the PlantVillage data, the model weights we release, and the pretrained
backbones we start from.

## 1. Code — GPL-2.0-only

The repository ships `LICENSE` (GNU GPL v2). Everything under `src/`, `scripts/`, `app/`, `tests/`
and the notebooks is covered by it.

## 2. Dataset — CC BY-SA 3.0

The original release states this explicitly (Hughes & Salathé 2015, *An open access repository of
images on plant health to enable the development of mobile disease diagnostics*, arXiv:1511.08060):

> […] openly accessible and released under the Creative Commons Attribution-ShareAlike 3.0 […] with
> the clarification that algorithms trained on the data fall under the same license. We chose this
> license to ensure that any disease diagnostic algorithm developed on the data can be made freely
> available to anyone.

The Hugging Face mirror we download from (`mohanty/PlantVillage`) carries the same `cc-by-sa-3.0`
tag. Attribution requirements are met through `CITATION.md`, the model card, and this file.

An earlier version of `DATASET_NOTES.md` in this repository said CC BY-NC-SA 3.0. That was wrong:
there is no non-commercial clause on the data. (The 2016 Frontiers *paper* is CC BY-NC-SA 4.0, which
is probably where the confusion came from — a paper's licence is not the data's licence.)

## 3. Model weights — CC BY-SA 3.0

Because the paper's clarification puts trained algorithms under the same licence, the checkpoints in
`models/exported/` and on the Hugging Face Hub are released under **CC BY-SA 3.0**, with attribution
to the PlantVillage authors. Anyone redistributing them or a fine-tuned derivative must keep the
same licence and the attribution.

This is a genuine constraint, not boilerplate: CC BY-SA 3.0 is a copyleft licence with no
non-commercial restriction but with a share-alike obligation.

## 4. Pretrained backbones — read this before commercial use

We fine-tune `torchvision` MobileNetV2 and EfficientNet-B0 weights. The torchvision *code* is
BSD-3-Clause, but the *weights* were trained on ImageNet, whose terms of access grant use for
non-commercial research and educational purposes. PyTorch does not attach a separate licence to the
published weights, and the question of whether ImageNet's terms reach through to a fine-tuned model
has no settled answer.

For this project — research and teaching — that is fine. Anyone planning commercial use should
either get their own legal reading or retrain the backbone from weights with clear commercial terms.

## 5. Split manifests

`data/splits/manifest.csv` is metadata derived from PlantVillage file names (paths, class names,
leaf groups, checksums). It contains no image data. We distribute it under CC BY-SA 3.0 as a
derivative of the dataset. Raw images are never committed; `src/data/prepare.py` downloads them from
the pinned upstream revision.

## 6. Example images in the demo

The Gradio Space built by `scripts/build_hf_release.py` includes a handful of PlantVillage test
images as clickable examples. They are redistributed under CC BY-SA 3.0 with attribution in the
Space README, which the licence permits.

## Release gate

| Check | Status |
|---|---|
| Dataset licence identified from a primary source | Done — CC BY-SA 3.0 (quote above) |
| Public release of weights permitted | Yes, under CC BY-SA 3.0 with attribution |
| Redistribution of split manifests permitted | Yes, under CC BY-SA 3.0 |
| Hugging Face hosting permitted | Yes — same conditions; model card states the licence and attribution |
| Code licence compatible with the data licence | Yes — separate works distributed together (aggregation), not a merged derivative: GPL-2.0 code processes CC BY-SA data, and neither licence's copyleft reaches the other |
| Third-party library licences checked | PyTorch/torchvision BSD-3, NumPy/pandas/scikit-learn/matplotlib BSD-3, Pillow MIT-CMU, PyYAML MIT, tqdm MPL-2.0/MIT, safetensors Apache-2.0, Gradio Apache-2.0 — all compatible with distribution here |
| Backbone weights caveat documented | Yes, section 4 |

This file replaces the earlier `LICENSE_PLACEHOLDER.md`.
