"""Can the class be read off the background alone?

    python scripts/background_probe.py

Motivation: the models lose accuracy when the background is replaced, mostly on the
Tomato classes. If the background by itself predicts the class far above chance, then
part of what a classifier learns on PlantVillage is which photo session an image came
from, not what the leaf looks like.

Method: take the dataset's segmented copy of each image as a leaf mask, keep only the
*background* pixels of the colour photo, reduce them to simple colour statistics
(per-channel mean, standard deviation and a small hue/value histogram) and fit
multinomial logistic regression on the training split. Then score it on the test split
— the same leaf-level split the models use, so this probe cannot cheat either.

The leaf is never shown to the probe, so its accuracy is a lower bound on how much
class information the background carries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import load_split  # noqa: E402
from src.data.labels import PLANTVILLAGE_CLASSES, pretty_name  # noqa: E402

IMAGE_ROOT = REPO / "data" / "raw" / "plantvillage"


def background_features(color_path: Path, segmented_path: Path, size: int = 64) -> np.ndarray | None:
    with Image.open(color_path) as im:
        color = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255
    with Image.open(segmented_path) as im:
        seg = np.asarray(im.convert("RGB").resize((size, size), Image.NEAREST), dtype=np.int16)

    background = color[seg.sum(axis=2) <= 40]
    if len(background) < 200:  # leaf fills the frame, nothing to measure
        return None
    mean, std = background.mean(axis=0), background.std(axis=0)
    brightness = background.mean(axis=1)
    histogram = np.histogram(brightness, bins=8, range=(0, 1), density=True)[0]
    return np.concatenate([mean, std, histogram, [background.mean(), background.std()]])


def collect(df: pd.DataFrame, per_class: int, seed: int, desc: str) -> tuple[np.ndarray, np.ndarray]:
    counts = df["class_name"].value_counts()
    sample = df.groupby("class_name").sample(per_class, random_state=seed, replace=False) \
        if counts.min() >= per_class else pd.concat(
            [g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby("class_name")]
        )
    features, labels = [], []
    for path, segmented, class_name in tqdm(
        list(zip(sample["path"], sample["segmented_path"], sample["class_name"])), desc=desc
    ):
        vector = background_features(IMAGE_ROOT / path, IMAGE_ROOT / segmented)
        if vector is not None:
            features.append(vector)
            labels.append(PLANTVILLAGE_CLASSES.index(class_name))
    return np.stack(features), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-per-class", type=int, default=60)
    parser.add_argument("--test-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = load_split(REPO / "data" / "splits" / "manifest.csv")
    manifest = manifest[manifest["segmented_path"].notna()]
    train = manifest[manifest["split"] == "train"]
    test = manifest[manifest["split"] == "test"]

    x_train, y_train = collect(train, args.train_per_class, args.seed, "train backgrounds")
    x_test, y_test = collect(test, args.test_per_class, args.seed, "test backgrounds")

    scaler = StandardScaler().fit(x_train)
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(scaler.transform(x_train), y_train)
    predictions = probe.predict(scaler.transform(x_test))

    accuracy = float((predictions == y_test).mean())
    chance = 1 / len(PLANTVILLAGE_CLASSES)
    crops = np.array([c.split("___")[0] for c in PLANTVILLAGE_CLASSES])
    crop_accuracy = float((crops[predictions] == crops[y_test]).mean())

    per_class = (
        pd.DataFrame({"true": y_test, "hit": predictions == y_test})
        .groupby("true")["hit"].agg(["mean", "size"])
        .rename(index=lambda i: PLANTVILLAGE_CLASSES[i])
        .sort_values("mean", ascending=False)
    )

    print(f"\nBackground-only probe on {len(y_test)} held-out images, {len(y_train)} training images")
    print(f"  class accuracy   {accuracy:.3f}   (chance {chance:.3f}, {accuracy / chance:.0f}x)")
    print(f"  crop accuracy    {crop_accuracy:.3f}   (14 crops)")
    print("\n  best-recognised classes from background alone:")
    for name, row in per_class.head(8).iterrows():
        print(f"    {pretty_name(name):55s} {row['mean']:.2f}  (n={int(row['size'])})")

    out = REPO / "artifacts" / "reports" / "background_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_train": len(y_train), "n_test": len(y_test),
            "class_accuracy": accuracy, "chance": chance, "crop_accuracy": crop_accuracy,
            "per_class_accuracy": {PLANTVILLAGE_CLASSES[i]: float(v)
                                   for i, v in per_class["mean"].rename(index=PLANTVILLAGE_CLASSES.index).items()},
        }, f, indent=2)
    print(f"\nWrote {out.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
