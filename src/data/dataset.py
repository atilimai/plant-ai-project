from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.labels import class_names_for, target_for
from src.data.transforms import build_transforms
from src.utils.seed import seed_worker

# Median background colour of the training photos, measured over 300 images using their
# segmented masks. Used by the "segmented_gray" variant.
BACKGROUND_RGB = (140, 132, 136)
VARIANTS = ("color", "segmented", "segmented_gray")


class PlantVillageDataset(Dataset):
    """Reads images listed in a split manifest.

    ``variant`` picks which copy of each photo to read:

    * ``color`` — the original photograph (what we train on),
    * ``segmented`` — the dataset's background-removed copy, black outside the leaf,
    * ``segmented_gray`` — the same leaf composited onto a flat background in the
      colour the lab table actually has.

    The last two are only used for the robustness check: ``segmented`` changes both
    the background *content* and its appearance, while ``segmented_gray`` keeps the
    appearance plausible, which separates "the model reads the background" from
    "a black background is simply out of distribution". Rows without a segmented
    counterpart are dropped in those modes.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: str | Path,
        task: str = "multiclass",
        transform: Callable | None = None,
        variant: str = "color",
    ):
        self.image_root = Path(image_root)
        self.task = task
        self.transform = transform
        self.variant = variant
        self.class_names = class_names_for(task)

        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}, expected one of {VARIANTS}")
        column = "path" if variant == "color" else "segmented_path"
        if column not in manifest:
            raise KeyError(f"Manifest has no {column!r} column for variant {variant!r}")
        df = manifest[manifest[column].notna()].reset_index(drop=True)
        self.paths = df[column].tolist()
        self.source_paths = df["path"].tolist()
        self.targets = [target_for(c, task) for c in df["class_name"]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.image_root / self.paths[idx]) as im:
            image = im.convert("RGB")
        if self.variant == "segmented_gray":
            image = paste_on_background(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]

    def class_counts(self) -> torch.Tensor:
        return torch.bincount(torch.tensor(self.targets), minlength=len(self.class_names))


def paste_on_background(segmented: Image.Image, rgb: tuple[int, int, int] = BACKGROUND_RGB,
                         threshold: int = 40) -> Image.Image:
    """Replace the black background of a segmented image with a flat colour."""
    arr = np.asarray(segmented.convert("RGB"))
    out = np.where((arr.sum(axis=2, keepdims=True) > threshold), arr, np.array(rgb, dtype=arr.dtype))
    return Image.fromarray(out.astype(np.uint8))


def load_split(manifest_path: str | Path, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if "has_segmented" in df:
        stems = df["path"].str.split("/").str[1] + "/" + df["path"].map(lambda p: Path(p).stem)
        df["segmented_path"] = ("segmented/" + stems + "_final_masked.jpg").where(df["has_segmented"])
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def make_loader(
    manifest: pd.DataFrame,
    image_root: str | Path,
    task: str,
    train: bool,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    aug: dict | None = None,
    variant: str = "color",
    seed: int = 0,
) -> DataLoader:
    ds = PlantVillageDataset(
        manifest, image_root, task=task, variant=variant,
        transform=build_transforms(train, image_size, aug),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
