"""Evaluate a checkpoint on a manifest split and write reports + figures.

    python -m src.evaluation.evaluate --checkpoint models/checkpoints/multiclass_mobilenet_v2/best.pt
    python -m src.evaluation.evaluate --checkpoint ... --variant segmented   # background robustness

Reports go to artifacts/reports/<run>/<split>[_<variant>]/, figures to
artifacts/figures/<run>/. Test-set evaluation refuses to run unless the committed
leakage audit passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.dataset import VARIANTS, load_split, make_loader  # noqa: E402
from src.data.labels import is_healthy  # noqa: E402
from src.evaluation.metrics import classification_metrics, confusion, top_confusions  # noqa: E402
from src.inference.predictor import load_checkpoint  # noqa: E402
from src.training.trainer import predict  # noqa: E402
from src.utils.config import display_path, resolve_path  # noqa: E402
from src.utils.device import pick_device  # noqa: E402
from src.visualization.confusion_matrix import plot_confusion_matrix  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_per_class_f1,
    plot_reliability,
    plot_training_curves,
)


def check_audit(splits_dir: Path) -> None:
    audit_path = splits_dir / "leakage_audit.json"
    if not audit_path.exists():
        raise RuntimeError(f"{audit_path} not found; run `python -m src.data.prepare` first")
    with open(audit_path, encoding="utf-8") as f:
        if not json.load(f)["passed"]:
            raise RuntimeError("Leakage audit failed; refusing to report test-set metrics")


def derived_binary(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    """Healthy-vs-diseased metrics read off a multiclass model, by summing class probabilities."""
    healthy = np.array([is_healthy(c) for c in class_names])
    p_diseased = probs[:, ~healthy].sum(axis=1)
    y_bin = (~healthy[y_true]).astype(int)
    probs_bin = np.stack([1 - p_diseased, p_diseased], axis=1)
    return classification_metrics(y_bin, probs_bin.argmax(1), ["healthy", "diseased"], probs=probs_bin)


def evaluate(
    checkpoint: str | Path,
    split: str = "test",
    variant: str = "color",
    manifest: str | Path = "data/splits/manifest.csv",
    image_root: str | Path = "data/raw/plantvillage",
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "auto",
    reports_root: str | Path = "artifacts/reports",
    figures_root: str | Path = "artifacts/figures",
) -> dict:
    manifest = resolve_path(manifest)
    if split == "test":
        check_audit(manifest.parent)

    dev = pick_device(device)
    checkpoint = resolve_path(checkpoint)
    model, meta = load_checkpoint(checkpoint, dev)
    run_name = checkpoint.parent.name if checkpoint.is_file() else checkpoint.name
    class_names, task = meta["class_names"], meta["task"]

    df = load_split(manifest, split)
    loader = make_loader(df, resolve_path(image_root), task, train=False, image_size=meta["image_size"],
                         batch_size=batch_size, num_workers=num_workers, variant=variant)
    y_true, probs, loss = predict(model, loader, dev, amp=dev.type == "cuda", desc=f"{run_name}:{split}:{variant}")
    y_pred = probs.argmax(axis=1)

    tag = split if variant == "color" else f"{split}_{variant}"
    report_dir = resolve_path(reports_root) / run_name / tag
    figure_dir = resolve_path(figures_root) / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = classification_metrics(y_true, y_pred, class_names, probs=probs)
    metrics.update({"run": run_name, "task": task, "arch": meta["arch"], "split": split,
                    "variant": variant, "loss": loss})
    cm = confusion(y_true, y_pred, len(class_names))
    metrics["top_confusions"] = top_confusions(cm, class_names, k=15)
    if task == "multiclass":
        metrics["derived_binary"] = {k: v for k, v in derived_binary(y_true, probs, class_names).items()
                                     if k != "per_class"}

    ds = loader.dataset
    order = np.argsort(-probs, axis=1)
    preds = pd.DataFrame({
        "path": ds.source_paths,
        "true": [class_names[i] for i in y_true],
        "predicted": [class_names[i] for i in y_pred],
        "confidence": probs.max(axis=1),
        "p_true": probs[np.arange(len(y_true)), y_true],
        "correct": y_true == y_pred,
        "second": [class_names[i] for i in order[:, 1]],
    })
    if split == "test" and variant == "color":
        # Per-image outputs are the evidence behind the headline numbers and what the
        # failure analysis reads. For validation and for the background variants the
        # aggregate metrics are enough; re-run this script if you need the rest.
        preds.to_csv(report_dir / "predictions.csv", index=False, float_format="%.5f")
        np.save(report_dir / "probabilities.npy", probs.astype(np.float16))
    pd.DataFrame(metrics["per_class"]).to_csv(report_dir / "per_class_metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(report_dir / "confusion_matrix.csv")
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    nice = f"{run_name} · {split}{'' if variant == 'color' else f' ({variant})'}"
    plot_confusion_matrix(cm, class_names, normalize=True, title=f"{nice} · row-normalised",
                          save_path=figure_dir / f"confusion_matrix_{tag}_normalized.png")
    plot_confusion_matrix(cm, class_names, normalize=False, title=f"{nice} · counts",
                          save_path=figure_dir / f"confusion_matrix_{tag}_counts.png")
    plot_per_class_f1(metrics["per_class"], title=f"{nice} · per-class F1",
                      save_path=figure_dir / f"per_class_f1_{tag}.png")
    plot_reliability(probs, y_true, title=f"{nice} · ECE {metrics['ece']:.3f}",
                     save_path=figure_dir / f"reliability_{tag}.png")
    history = checkpoint.parent / "history.csv"
    if checkpoint.is_file() and history.exists():
        plot_training_curves(pd.read_csv(history), title=run_name, save_path=figure_dir / "training_curves.png")
    plt.close("all")

    headline = f"acc {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | bal-acc {metrics['balanced_accuracy']:.4f}"
    print(f"{nice}: {metrics['n_samples']} images | {headline} -> {display_path(report_dir)}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="color", choices=list(VARIANTS))
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--image-root", default="data/raw/plantvillage")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    evaluate(args.checkpoint, args.split, args.variant, args.manifest, args.image_root,
             args.batch_size, args.num_workers, args.device)


if __name__ == "__main__":
    main()
