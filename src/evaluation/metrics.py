from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top_k = np.argsort(-probs, axis=1)[:, :k]
    return float((top_k == y_true[:, None]).any(axis=1).mean())


def classification_metrics(
    y_true, y_pred, class_names: list[str], probs: np.ndarray | None = None
) -> dict:
    """Everything we report for a split, as plain Python types (JSON-ready)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = {
        "n_samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        # Mean recall over the classes present in y_true (same as sklearn's
        # balanced_accuracy_score, minus its warning when a prediction hits an absent class).
        "balanced_accuracy": float(r[support > 0].mean()),
    }
    for avg in ("macro", "weighted"):
        ap, ar, af, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        out[f"{avg}_precision"] = float(ap)
        out[f"{avg}_recall"] = float(ar)
        out[f"{avg}_f1"] = float(af)

    out["per_class"] = [
        {
            "class": name,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    ]

    if probs is not None:
        probs = np.asarray(probs)
        out["ece"] = expected_calibration_error(probs, y_true)
        if len(class_names) > 2:
            for k in (3, 5):
                out[f"top{k}_accuracy"] = top_k_accuracy(probs, y_true, k)
        elif len(np.unique(y_true)) == 2:
            # Binary track: class 1 is "diseased", the class we care about catching.
            out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            out["average_precision"] = float(average_precision_score(y_true, probs[:, 1]))

    if len(class_names) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if tp + fn else 0.0
        out["specificity"] = float(tn / (tn + fp)) if tn + fp else 0.0
        out["false_negatives"] = int(fn)
        out["false_positives"] = int(fp)
    return out


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def top_confusions(cm: np.ndarray, class_names: list[str], k: int = 10) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:k]
    rows = []
    for flat in order:
        i, j = np.unravel_index(flat, off.shape)
        if off[i, j] == 0:
            break
        rows.append({
            "true": class_names[i],
            "predicted": class_names[j],
            "count": int(off[i, j]),
            "share_of_true_class": float(off[i, j] / cm[i].sum()),
        })
    return rows
