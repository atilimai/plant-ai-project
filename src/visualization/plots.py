"""Smaller report figures: training curves, per-class F1, reliability diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.labels import pretty_name


def save_figure(fig, save_path, dpi: int = 150):
    """Write a figure, letting the suffix decide the format.

    Charts stay PNG; figures made of photographs (galleries, Grad-CAM grids) are saved
    as .jpg, which is roughly ten times smaller for the same visual quality and keeps
    the committed artifacts reviewable in git.
    """
    if not save_path:
        return fig
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"pil_kwargs": {"quality": 88, "optimize": True}} if path.suffix in {".jpg", ".jpeg"} else {}
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    return fig


def _save(fig, save_path):
    return save_figure(fig, save_path)


def plot_training_curves(history: pd.DataFrame, title: str | None = None, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ep = history["epoch"]
    axes[0].plot(ep, history["train_loss"], marker="o", label="train")
    axes[0].plot(ep, history["val_loss"], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[1].plot(ep, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(ep, history["val_accuracy"], marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[2].plot(ep, history["val_macro_f1"], marker="o", color="C2", label="val macro-F1")
    axes[2].plot(ep, history["val_balanced_accuracy"], marker="s", color="C3", label="val balanced acc.")
    axes[2].set_title("Validation")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    # Train metrics include augmentation and label smoothing, so train loss sitting
    # above val loss is expected here and not a bug.
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_per_class_f1(per_class: list[dict], title: str | None = None, save_path=None):
    df = pd.DataFrame(per_class).sort_values("f1")
    labels = [pretty_name(c) if "___" in c else c for c in df["class"]]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.24 * len(df) + 1)))
    colors = ["C3" if f < 0.9 else "C0" for f in df["f1"]]
    ax.barh(labels, df["f1"], color=colors)
    for y, (f1, n) in enumerate(zip(df["f1"], df["support"])):
        ax.text(min(f1, 1.0) + 0.005, y, f"{f1:.3f}  (n={n})", va="center", fontsize=7)
    ax.set_xlim(max(0.0, df["f1"].min() - 0.1), 1.08)
    ax.set_xlabel("F1 on the held-out split")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15, title: str | None = None, save_path=None):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            centers.append(confidence[m].mean())
            accs.append(correct[m].mean())
            counts.append(m.sum())

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    ax.plot(centers, accs, marker="o", label="model")
    for c, a, n in zip(centers, accs, counts):
        ax.annotate(str(n), (c, a), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, save_path)
