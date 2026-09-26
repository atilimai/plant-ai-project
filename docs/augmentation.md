# Augmentation strategy

Implemented in `src/data/transforms.py`, configured under `augmentation:` in
`configs/train_base.yaml`, and applied to the **training split only** — validation and test images
are resized and normalised, nothing else.

## What we apply

| Transform | Setting | Why |
|---|---|---|
| `RandomResizedCrop` | 224 px, scale 0.6–1.0 | Scale and position invariance. The lower bound stays at 0.6 because lesions can be small: an aggressive crop can remove the only symptom and effectively mislabel the image. |
| Horizontal + vertical flip | p = 0.5 each | A detached leaf on a table has no canonical orientation. |
| 90° rotations | p = 0.75 (one of three non-zero angles) | Same reason, and unlike free rotation it needs no padding. |
| `ColorJitter` | brightness 0.2, contrast 0.2, saturation 0.2, hue 0.02, applied with p = 0.8 | Robustness to lighting and white balance. |
| Normalisation | ImageNet mean/std | The backbones were pretrained with it. |

## What we deliberately do not apply

* **Free-angle rotation.** `RandomRotation(±30)` fills the corners with a constant colour. On these
  uniform backgrounds that is a visible artefact correlated with nothing, and the network can
  learn to key on it. Right-angle rotations give the same invariance for free.
* **Strong hue shifts.** Disease signal *is* colour: chlorosis (yellowing), rust (orange-brown),
  necrosis (brown-black). A ±0.1 hue jitter can turn a healthy leaf into something that looks
  chlorotic. We keep hue at ±0.02.
* **Grayscale conversion.** Same reason.
* **Aggressive erasing / cutout.** A random box can cover the lesion, which is exactly the region
  the label depends on.
* **Mixup / CutMix.** They help on large, noisy datasets; here they would blend two disease classes
  into an image whose label is a mixture, and our main interest is honest per-class metrics on a
  small, clean dataset.
* **Background replacement.** Tempting as a way to attack the lab-background bias, but PlantVillage
  segmentation masks are imperfect and the pasted result is visibly artificial. The honest answer
  to background bias is field data, which this project does not have. We measure the size of the
  problem instead: see the background-removed evaluation in `docs/results.md`.

## Sanity check

`notebooks/00_dataset_inspection.ipynb` shows five augmented versions of four training images next
to the originals. The check we care about is simple: after augmentation, can a person still see the
symptom the label refers to? If not, the augmentation is too strong.

## Effect

Augmentation is on in every run in `configs/`. The training curves in
`artifacts/figures/<run>/training_curves.png` show train loss sitting above validation loss, which
is expected: training loss is computed on augmented images with label smoothing, validation loss on
clean ones.
