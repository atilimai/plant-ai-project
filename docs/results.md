# Results

All numbers come from the held-out test split defined in
[`data/splits/README.md`](../data/splits/README.md): 7,770 images, no leaf shared with training,
audited. Raw outputs are in `artifacts/reports/`; this page is the reading of them.

Reproduce with `python scripts/run_experiments.py`.

## Headline

| Model | Accuracy | Balanced acc. | Macro-F1 | Weighted F1 | Top-3 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 · 38 classes | 0.9932 | 0.9922 | 0.9905 | 0.9932 | 0.9996 | 0.142 |
| EfficientNet-B0 · 38 classes | 0.9952 | 0.9939 | 0.9938 | 0.9952 | 0.9992 | 0.143 |
| MobileNetV2 · healthy vs diseased | 0.9986 | 0.9982 | 0.9982 | 0.9986 | – | 0.064 |
| EfficientNet-B0 · healthy vs diseased | 0.9985 | 0.9981 | 0.9981 | 0.9985 | – | 0.066 |

53 errors out of 7,770 test images for MobileNetV2, 37 for EfficientNet-B0.

Training cost: 11–24 minutes per run on one RTX 4060 Laptop GPU
(13, 15, 9 and 15 epochs; early stopping on validation macro-F1).

## How to read these numbers

Two things are worth saying before the details.

**They are not comparable with published PlantVillage results.** Papers usually report 99%+ on
image-level splits, where photos of the same physical leaf sit on both sides. Our split keeps every
leaf on one side, which is a harder and more meaningful test. Accuracy stays high anyway — which
says the task is genuinely easy *in lab conditions*, not that the leakage did not matter. Measured
on images with known leaf ids, a random image split leaves 99.8% of test images with a sibling in
training; a model can do very well on such a test set without generalising at all.

**They describe lab photographs, not fields.** Every image is one detached leaf on a uniform
background. The next section is our attempt to measure how much of the performance depends on that
setting rather than on the leaf itself.

## What the models actually use: three experiments

### 1. Change the background

The dataset ships a background-removed copy of almost every photo, so the same test images can be
re-run with the background changed and nothing else (7,611 of 7,770 images have such a copy):

| Test images | MobileNetV2 38-class | EfficientNet-B0 38-class | MobileNetV2 binary | EfficientNet-B0 binary |
|---|---:|---:|---:|---:|
| Original photographs | 0.9932 | 0.9952 | 0.9986 | 0.9985 |
| Leaf on a flat lab-coloured background | 0.8649 | 0.8430 | 0.9656 | 0.9704 |
| Leaf on black (dataset's segmented copy) | 0.7259 | 0.8028 | 0.9762 | 0.9735 |

The middle row is ours: the segmented leaf composited onto a flat background in the colour the lab
table actually has (RGB 140/132/136, measured over 300 training photos). It exists to separate two
different things that removing a background does — losing the information the background carried,
and showing the network an image unlike anything it was trained on.

Read across the rows:

* **Disease identification loses 13–15 points** the moment the background is replaced by a plausible
  flat colour. The leaf is untouched, so that gap is the part of the decision that was not coming
  from the leaf.
* **A black frame costs more on top of that** — dramatically so for MobileNetV2 (another 14 points),
  mildly for EfficientNet-B0 (4 points). That extra is the out-of-distribution shock, and it is
  architecture-dependent rather than a property of the task.
* **The binary task barely moves** (2–3 points). Whether a leaf is diseased is written on the leaf;
  which of 38 classes it belongs to apparently is not, entirely.

### 2. Ask whether the background alone predicts the class

If the background carries class information, a classifier should be able to read the class off the
background *by itself*. `scripts/background_probe.py` masks out the leaf, reduces the remaining
pixels to 16 colour statistics and fits logistic regression on the training split, then scores it on
the same leaf-level test split:

| Probe (leaf never shown) | Result |
|---|---:|
| Class accuracy over 38 classes | **33.5%** (chance 2.6%, 13×) |
| Crop accuracy over 14 crops | 43.9% |
| Best single class (Potato early blight) | 80% |

Sixteen numbers describing only the table behind the leaf identify the class a third of the time.
Images of one class were largely shot in one session, so lighting, surface and framing correlate
with the label. A model is free to use that shortcut, and the experiment above shows ours partly
does.

### 3. Look at where the attention goes

Grad-CAM on the last convolutional block, with the segmented mask used to measure how much of the
heatmap mass falls on the leaf (the leaf covers ~47% of the frame, so a uniform map would score
~0.47 and a ratio of 1.0):

| Model | CAM mass on leaf (correct) | Focus ratio | CAM mass on leaf (errors) | Focus ratio |
|---|---:|---:|---:|---:|
| MobileNetV2 · 38 classes | 0.903 | 2.03 | 0.795 | 2.02 |
| EfficientNet-B0 · 38 classes | 0.812 | 1.89 | 0.720 | 1.78 |
| MobileNetV2 · binary | 0.848 | 1.90 | 0.780 | 1.71 |
| EfficientNet-B0 · binary | 0.699 | 1.57 | 0.765 | 1.69 |

This is the interesting part. Attention sits firmly on the leaf — about twice what a uniform
heatmap would give — and yet changing the background still costs 13–15 points on the 38-class task.
Grad-CAM answers *where the evidence for this prediction is*, not *what the model would do without
the rest of the image*; experiments 1 and 2 answer that, and they disagree with the comfortable
reading of the heatmaps. A clean saliency map is not evidence of robustness.

## Per-class behaviour (MobileNetV2, 38 classes)

31 of 38 classes reach F1 ≥ 0.98. The weakest:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Corn – Cercospora leaf spot / Gray leaf spot | 56 | 0.930 | 0.946 | 0.938 |
| Potato – Healthy | 24 | 0.889 | 1.000 | 0.941 |
| Corn – Northern leaf blight | 124 | 0.967 | 0.960 | 0.964 |
| Tomato – Early blight | 145 | 0.959 | 0.972 | 0.966 |
| Tomato – Late blight | 252 | 0.965 | 0.984 | 0.974 |
| Tomato – Bacterial spot | 312 | 0.974 | 0.978 | 0.976 |

Class size is not what drives this: the rank correlation between support and F1 is 0.02, and
several of the smallest classes (Tomato mosaic virus with 37 test images, Grape healthy with 43,
Apple cedar rust with 44) are classified perfectly. What drives it is visual similarity within a crop: the two
Corn lesion classes (Cercospora leaf spot and Northern leaf blight) are elongated grey-brown
lesions on the same leaf type and account for the largest confusion after the tomato cluster.

## Errors

| | MobileNetV2 38-class | EfficientNet-B0 38-class |
|---|---:|---:|
| Errors | 53 / 7,770 (0.68%) | 37 / 7,770 (0.48%) |
| Errors above 0.9 confidence | 0 | 2 |
| Within the same crop | 46 | 33 |
| Across crops | 7 | 4 |
| Healthy called diseased | 4 | 1 |
| Diseased called healthy | 5 | 1 |

Errors are overwhelmingly *within* a crop: the models essentially never mistake a tomato for a
grape, they mistake one tomato disease for another. That matches the top confusions — Tomato yellow
leaf curl virus → bacterial spot (7), Corn northern leaf blight → Cercospora (4) — and it is the
practically relevant failure mode, since crop identity is usually known to the user anyway.

Nine images cross the healthy/diseased line in the 38-class model, five of them diseased leaves
called healthy. Those are the expensive mistakes, which is why the binary track is reported
separately below.

Per-run breakdowns, including error rate by crop: `artifacts/reports/<run>/failure_analysis.md`.

## Calibration

ECE looks poor (0.14 on the 38-class models), but in the harmless direction: mean confidence is
0.851 against 0.993 accuracy, i.e. the models are systematically **under**confident. That is what
label smoothing 0.1 does — it caps the target probability at 0.9 — and it does not damage the
ranking:

| Confidence band | Images | Accuracy |
|---|---:|---:|
| < 0.5 | 63 | 0.571 |
| 0.5 – 0.7 | 98 | 0.867 |
| 0.7 – 0.9 | 5,513 | 0.998 |
| ≥ 0.9 | 2,096 | 1.000 |

So the score is usable as a triage signal even though it is not a probability: accepting only
predictions above 0.7 covers 98% of the test set at 99.8% accuracy. If calibrated probabilities are
needed, temperature scaling on the validation split is the standard fix and is listed in
`ROADMAP.md`. Reliability diagrams: `artifacts/figures/<run>/reliability_test.png`.

## Binary track, and whether it is needed

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC | Missed diseased | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 (dedicated) | 0.9986 | 0.9991 | 0.9972 | 0.9999 | 5 | 6 |
| EfficientNet-B0 (dedicated) | 0.9985 | 0.9989 | 0.9972 | 1.0000 | 6 | 6 |
| MobileNetV2 38-class, probabilities summed | 0.9988 | 0.9993 | 0.9977 | 0.9999 | 4 | 5 |
| EfficientNet-B0 38-class, probabilities summed | **0.9996** | 0.9998 | 0.9991 | 0.9999 | **1** | 2 |

Training a dedicated binary model buys nothing: summing the disease probabilities of the 38-class
model is as good or better, and EfficientNet-B0 gets there with a single missed diseased leaf out of
5,621. For deployment that means one model, two views — which is how the demo app is wired.

## Architecture comparison

EfficientNet-B0 is better on the 38-class task (99.52% vs 99.32%, macro-F1 0.9938 vs 0.9905) and
degrades less when the background goes black, at roughly 1.8× the parameters (4.1M vs 2.3M) and
1.5× the training time. MobileNetV2 is the better default if the model is ever going to run on a
phone; EfficientNet-B0 is the one to publish as the strongest result. Both are far past the point
where the architecture is what limits this benchmark.

## Limitations of this evaluation

* **One split, no error bars.** Classes with few test images (Potato healthy: 24, Tomato mosaic
  virus: 37, Grape healthy: 43) have per-class F1 with wide uncertainty. Leaf-grouped
  cross-validation would fix this and is listed in `ROADMAP.md`.
* **Classes without author leaf ids are split by camera sequence**, so their test images come from
  contiguous shooting segments: a stricter test (a whole session is held out), but a less random
  one, and 1,008 held-out images were dropped by the purge buffer to keep it leak-free.
* **Segmented copies are imperfect.** Masks occasionally clip leaf edges and always remove the
  shadow, so the background experiments are an upper bound on the damage from losing the
  background — but the background-only probe is independent evidence, and it agrees.
* **No field images.** Nothing here supports a claim about real-world use. Given what experiment 1
  shows about sensitivity to the surroundings, the expectation for field photos should be low until
  measured.
