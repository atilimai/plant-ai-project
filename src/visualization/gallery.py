from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.labels import pretty_name
from src.visualization.plots import save_figure


def _label(name: str) -> str:
    return pretty_name(name) if "___" in name else name


def pick_gallery_rows(predictions: pd.DataFrame, n_correct: int = 10, n_borderline: int = 5,
                      n_errors: int = 9, seed: int = 0) -> pd.DataFrame:
    """Confidence-stratified sample, so the gallery is not a highlight reel.

    * correct: confident correct predictions, at most one per class,
    * borderline: correct but with confidence below 0.7 (or the least confident ones),
    * errors: the most confident mistakes first, one per true class where possible.
    """
    correct = predictions[predictions["correct"]]
    wrong = predictions[~predictions["correct"]]

    confident = correct[correct["confidence"] >= 0.9].groupby("true").sample(1, random_state=seed)
    confident = confident.sample(min(n_correct, len(confident)), random_state=seed)
    if len(confident) < n_correct:  # weak model or few classes: top up with the most confident hits
        rest = correct.drop(confident.index).nlargest(n_correct - len(confident), "confidence")
        confident = pd.concat([confident, rest])

    correct = correct.drop(confident.index)
    borderline = correct[correct["confidence"] < 0.7]
    if len(borderline) < n_borderline:
        borderline = correct.nsmallest(n_borderline, "confidence")
    borderline = borderline.sample(min(n_borderline, len(borderline)), random_state=seed)

    errors = wrong.sort_values("confidence", ascending=False).drop_duplicates("true").head(n_errors)
    if len(errors) < n_errors:
        rest = wrong.drop(errors.index).sort_values("confidence", ascending=False)
        errors = pd.concat([errors, rest.head(n_errors - len(errors))])

    return pd.concat([
        confident.assign(kind="correct"),
        borderline.assign(kind="borderline"),
        errors.assign(kind="error"),
    ])


def plot_prediction_gallery(rows: pd.DataFrame, image_root: str | Path, n_cols: int = 6,
                            title: str | None = None, save_path: str | Path | None = None):
    image_root = Path(image_root)
    n_rows = int(np.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.9 * n_cols, 3.5 * n_rows))
    colors = {"correct": "#1a7f37", "borderline": "#9a6700", "error": "#cf222e"}

    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        with Image.open(image_root / row["path"]) as im:
            ax.imshow(im.convert("RGB"))
        caption = (
            f"pred: {_label(row['predicted'])}\n"
            f"true: {_label(row['true'])}\n"
            f"conf {row['confidence']:.2f} · {row['kind']}"
        )
        caption = "\n".join(textwrap.shorten(line, 38, placeholder="…") for line in caption.split("\n"))
        ax.set_title(caption, fontsize=8, color=colors[row["kind"]], loc="left")
        ax.axis("off")
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, save_path)
