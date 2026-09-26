from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.labels import pretty_name
from src.visualization.plots import save_figure


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Row-normalised (recall per true class) or raw-count confusion matrix.

    For the 38-class track only non-zero off-diagonal cells are annotated, otherwise
    the figure turns into a wall of zeros.
    """
    n = len(class_names)
    counts = cm.astype(int)
    values = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else counts
    labels = [pretty_name(c) if "___" in c else c for c in class_names]

    size = max(4.5, 0.36 * n + 2)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="share of true class" if normalize else "images")

    ax.set_xticks(range(n), labels, rotation=90 if n > 2 else 0, fontsize=7 if n > 10 else 10)
    ax.set_yticks(range(n), labels, fontsize=7 if n > 10 else 10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    dense = n <= 10
    threshold = values.max() / 2
    for i in range(n):
        for j in range(n):
            if not dense and (counts[i, j] == 0 or i == j):
                continue
            text = f"{values[i, j]:.2f}" if normalize else f"{counts[i, j]}"
            if dense and normalize:
                text += f"\n({counts[i, j]})"
            ax.text(j, i, text, ha="center", va="center", fontsize=6 if n > 10 else 10,
                    color="white" if values[i, j] > threshold else "black")

    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return save_figure(fig, save_path)
