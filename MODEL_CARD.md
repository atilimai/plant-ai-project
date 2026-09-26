# Model card — PlantVillage leaf disease classifiers

Four models, one recipe: MobileNetV2 and EfficientNet-B0, each fine-tuned for a 38-class disease
task and for a binary healthy/diseased task, on a leaf-level, leakage-audited split of PlantVillage.

| | |
|---|---|
| Version | 1.0.0 |
| Framework | PyTorch (torchvision backbones) |
| Input | RGB image, resized to 224×224, ImageNet normalisation |
| Output | Class probabilities (38 classes, or 2 for the binary models) |
| Weights | `model.safetensors` + `config.json` (and ONNX) per run |
| Licence | CC BY-SA 3.0 (see [`LICENSING.md`](LICENSING.md)) |
| Repository | <https://github.com/atilimai/plant-ai-project> |
| Hub | <https://huggingface.co/atilimai/plantvillage-leaf-disease-classifier> |

## Intended use

Research, teaching and demonstrations on PlantVillage-like images: **a single detached leaf,
photographed against a plain background in even lighting**.

Out of scope, and unvalidated:

* photographs taken in the field (plants in soil, several leaves, sky, hands, shadows);
* crops or diseases outside the 38 classes — the model always returns one of them;
* any decision about treating a real crop. It is not a diagnostic device, and its errors are not
  distributed like an expert's.

## Training data

PlantVillage, colour variant: 54,305 images, 38 classes, 14 crops
([`DATASET_NOTES.md`](DATASET_NOTES.md)). Downloaded at a pinned revision with checksum
verification.

The split is made at the level of the **physical leaf**, not the image, because PlantVillage
contains several photos of each leaf. Measured on the images whose leaf identity the dataset
authors published, a random image-level split leaves 99.8% of test images with a sibling from the
same leaf in training; that is the difference between measuring generalisation and measuring
memorisation. Leaf ids were rebuilt from the authors' leaf map plus file-name evidence, and images
without an id are split by contiguous camera segments with a purge buffer. Policy, calibration of
that buffer and the audit: [`data/splits/README.md`](data/splits/README.md).

| Split | Images | Leaf groups |
|---|---:|---:|
| train | 38,011 | 13,996 |
| validation | 7,516 | 2,414 |
| test | 7,770 | 2,639 |
| excluded by the purge buffer | 1,008 | – |

## Training procedure

Identical for all four runs ([`configs/train_base.yaml`](configs/train_base.yaml)):

| | |
|---|---|
| Backbone | ImageNet-pretrained MobileNetV2 / EfficientNet-B0, classifier head replaced |
| Optimiser | AdamW, lr 1e-3 (head) and 3e-4 (backbone), weight decay 0.05, no decay on norms/biases |
| Schedule | 1 warm-up epoch, then cosine decay; up to 15 epochs, early stopping on validation macro-F1 (patience 4) |
| Batch size | 64, mixed precision on GPU |
| Loss | Cross-entropy, label smoothing 0.1, class weights ∝ 1/√frequency |
| Augmentation | RandomResizedCrop(0.6–1.0), horizontal/vertical flips, 90° rotations, mild colour jitter ([`docs/augmentation.md`](docs/augmentation.md)) |
| Hardware | One RTX 4060 Laptop GPU; about 20 minutes per run |

Model selection used the validation split only; the test split was touched once, at the end, by
`src.evaluation.evaluate` — which refuses to run unless the committed leakage audit passed.

## Evaluation

Held-out test split, 7,770 images:

| Model | Accuracy | Balanced acc. | Macro-F1 | Weighted F1 | Top-3 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 · 38 classes | 0.9932 | 0.9922 | 0.9905 | 0.9932 | 0.9996 | 0.142 |
| EfficientNet-B0 · 38 classes | 0.9952 | 0.9939 | 0.9938 | 0.9952 | 0.9992 | 0.143 |
| MobileNetV2 · healthy vs diseased | 0.9986 | 0.9982 | 0.9982 | 0.9986 | – | 0.064 |
| EfficientNet-B0 · healthy vs diseased | 0.9985 | 0.9981 | 0.9981 | 0.9985 | – | 0.066 |

31 of 38 classes reach F1 ≥ 0.98; the weakest are Corn Cercospora leaf spot (0.938) and Potato
healthy (0.941). Errors are almost always within a crop — one tomato disease mistaken for another —
rather than across crops. Full per-class tables:
`artifacts/reports/<run>/test/per_class_metrics.csv`. Discussion:
[`docs/results.md`](docs/results.md).

**Confidence is underconfident, not overconfident.** Label smoothing caps the target probability at
0.9, so mean confidence (0.851) sits below accuracy (0.993). The ranking is still useful: below 0.5
confidence accuracy is 57%, above 0.9 it is 100%, and a 0.7 threshold covers 98% of the test set at
99.8% accuracy. Treat the score as a triage signal, not a probability.

**A separate binary model is not needed.** Summing the disease probabilities of the 38-class model
matches or beats the dedicated binary models (EfficientNet-B0: 99.96% accuracy, one missed diseased
leaf out of 5,621, against 99.85% for the dedicated model).

### Robustness: the background matters more than it should

The same test photos, with the background replaced and the leaf untouched:

| Test images | MobileNetV2 38-class | EfficientNet-B0 38-class | binary models |
|---|---:|---:|---:|
| Original photographs | 0.9932 | 0.9952 | 0.9985–0.9986 |
| Leaf on a flat lab-coloured background | 0.8649 | 0.8430 | 0.9656–0.9704 |
| Leaf on black (dataset's segmented copy) | 0.7259 | 0.8028 | 0.9735–0.9762 |

Disease identification loses 13–15 points as soon as the background is replaced by a plausible flat
colour. A probe trained on background pixels alone (leaf masked out, 16 colour statistics) predicts
the class 33.5% of the time against a 2.6% chance rate, which explains why: classes were largely
photographed in single sessions, so the background carries label information that a model is free to
use. Grad-CAM shows attention concentrated on the leaf (about 2× a uniform map), which is a useful
reminder that saliency maps do not prove robustness.

The binary decision is much more robust (2–3 points), i.e. *whether* a leaf is diseased is written
on the leaf; *which* of 38 classes it is, partly is not.

## Limitations

* **Lab conditions only.** Single detached leaves, uniform background, even lighting. The
  background experiments above show the 38-class decision partly depends on those surroundings, so
  field performance should be assumed much lower until someone measures it. This repository contains
  no field data.
* **Closed set.** The model always answers with one of its classes. A photo of a crop, disease,
  pest or object outside the 38 classes gets a confident-looking wrong label, and confidence is not
  an out-of-distribution detector.
* **Small classes carry wide error bars.** Potato healthy has 24 test images, Tomato mosaic virus
  37, Grape healthy 43. A single split gives no confidence interval; leaf-grouped cross-validation
  is future work.
* **Nine of 7,770 test images cross the healthy/diseased line** in the 38-class MobileNetV2 model,
  five of them diseased leaves called healthy. Use the binary view (or the summed probabilities) if
  missing a diseased leaf is the costly error.
* **Class imbalance (36× between largest and smallest class)** is handled with class weights, and
  metrics are reported per class and macro-averaged so the aggregate cannot hide a weak class.
* **Resolution.** 224×224 input: symptoms smaller than a few pixels at that scale are simply not
  visible to the model.
* **Single-label.** Mixed infections, disease stage and severity are not represented in the data and
  cannot be predicted.
* **Not comparable to published PlantVillage numbers**, which are usually measured on image-level
  splits where the same leaf appears in train and test.

## Ethical and practical caveats

* **Not a diagnostic tool.** A wrong answer here can mean a wrongly sprayed field. Any real use
  needs an agronomist in the loop and validation on the data it will actually see.
* **Silent failure on out-of-distribution input.** The model always returns one of its classes with
  a confidence attached, including for a photo of a hand, a wall or a plant it has never seen.
  Confidence is not an abstention signal — the demo warns about this and shows Grad-CAM so a user
  can at least see whether the model looked at the leaf.
* **Uneven reliability across classes.** Classes with little data or unusual imagery are worse, and
  the per-class table should be read before trusting any single prediction.
* **Bias from the collection.** The dataset covers 14 crops, mostly from a handful of photo
  sessions. Cultivars, regions, growth stages and cameras outside that are not represented.
* **Share-alike licence.** Redistribution of the weights or a fine-tuned derivative must keep
  CC BY-SA 3.0 and credit the PlantVillage authors.

## How to use

```python
from huggingface_hub import snapshot_download
from src.inference.predictor import Predictor

model_dir = snapshot_download("atilimai/plantvillage-leaf-disease-classifier",
                              allow_patterns=["multiclass_mobilenet_v2/*"])
predictor = Predictor.load(f"{model_dir}/multiclass_mobilenet_v2")
print(predictor.predict("leaf.jpg", top_k=3)[0])
```

A dependency-light version (torch + torchvision only) is in `inference_example.py` in the Hub
repository. `predictor.explain(image)` additionally returns the Grad-CAM overlay.

## Citation

Cite the dataset papers (Mohanty et al. 2016; Hughes & Salathé 2015) — see
[`CITATION.md`](CITATION.md). Please state that numbers were obtained on a leaf-level split if you
compare them with other PlantVillage results.
