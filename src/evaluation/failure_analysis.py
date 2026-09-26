"""Failure analysis, prediction gallery and Grad-CAM figures for an evaluated run.

    python -m src.evaluation.failure_analysis --checkpoint models/checkpoints/multiclass_mobilenet_v2/best.pt

Run ``src.evaluation.evaluate`` on the test split first; this script reads its
predictions.csv. Besides error breakdowns it measures *where* the model looks:
PlantVillage ships a background-removed copy of most photos, which gives a free
leaf mask, so we can report the share of Grad-CAM mass that lands on the leaf.
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
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.data.dataset import load_split  # noqa: E402
from src.data.labels import is_healthy, pretty_name, split_class_name  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.inference.predictor import load_checkpoint  # noqa: E402
from src.models.factory import gradcam_layer  # noqa: E402
from src.utils.config import display_path, resolve_path  # noqa: E402
from src.utils.device import pick_device  # noqa: E402
from src.visualization.gallery import pick_gallery_rows, plot_prediction_gallery  # noqa: E402
from src.visualization.grad_cam import GradCAM, leaf_focus, plot_gradcam_triplet  # noqa: E402
from src.visualization.plots import save_figure  # noqa: E402


def leaf_mask(segmented_path: Path, size: int) -> np.ndarray:
    """Foreground of a segmented PlantVillage image (background is near-black)."""
    with Image.open(segmented_path) as im:
        arr = np.asarray(im.convert("RGB").resize((size, size), Image.NEAREST), dtype=np.int16)
    return (arr.sum(axis=2) > 40).astype(np.float32)


def error_breakdown(preds: pd.DataFrame, task: str) -> dict:
    wrong = preds[~preds["correct"]]
    out = {
        "n_images": int(len(preds)),
        "n_errors": int(len(wrong)),
        "error_rate": float(len(wrong) / len(preds)),
        "confident_errors_over_0.9": int((wrong["confidence"] > 0.9).sum()),
        "error_confidence_median": float(wrong["confidence"].median()) if len(wrong) else None,
        "correct_confidence_median": float(preds.loc[preds["correct"], "confidence"].median()),
    }
    if task != "multiclass":
        out["errors_by_true_class"] = wrong["true"].value_counts().to_dict()
        crops = preds["path"].str.split("/").str[1].map(lambda c: split_class_name(c)[0])
        out["error_rate_by_crop"] = (
            (~preds["correct"]).groupby(crops).mean().sort_values(ascending=False).round(4).to_dict()
        )
        return out

    true_crop = wrong["true"].map(lambda c: split_class_name(c)[0])
    pred_crop = wrong["predicted"].map(lambda c: split_class_name(c)[0])
    true_healthy = wrong["true"].map(is_healthy)
    pred_healthy = wrong["predicted"].map(is_healthy)
    out.update({
        "errors_within_same_crop": int((true_crop == pred_crop).sum()),
        "errors_across_crops": int((true_crop != pred_crop).sum()),
        "healthy_predicted_as_diseased": int((true_healthy & ~pred_healthy).sum()),
        "diseased_predicted_as_healthy": int((~true_healthy & pred_healthy).sum()),
        "disease_confused_with_other_disease": int((~true_healthy & ~pred_healthy).sum()),
    })
    crop_of = preds["true"].map(lambda c: split_class_name(c)[0])
    out["error_rate_by_crop"] = (~preds["correct"]).groupby(crop_of).mean().sort_values(ascending=False).round(4).to_dict()
    pairs = wrong.groupby(["true", "predicted"]).size().sort_values(ascending=False).head(15)
    out["top_confused_pairs"] = [{"true": t, "predicted": p, "count": int(n)} for (t, p), n in pairs.items()]
    return out


def gradcam_study(model, meta, rows: pd.DataFrame, image_root: Path, device, batch_size: int = 32):
    """Grad-CAM for the predicted class of each row; returns cams and leaf-focus stats."""
    size = meta["image_size"]
    tf = build_transforms(train=False, image_size=size)
    cams, stats = [], []
    with GradCAM(model, gradcam_layer(model)) as cam:
        for start in tqdm(range(0, len(rows), batch_size), desc="grad-cam", leave=False):
            chunk = rows.iloc[start:start + batch_size]
            images = []
            for p in chunk["path"]:
                with Image.open(image_root / p) as im:
                    images.append(tf(im.convert("RGB")))
            maps, _ = cam(torch.stack(images).to(device))
            for (_, row), m in zip(chunk.iterrows(), maps):
                cams.append(m)
                if isinstance(row.get("segmented_path"), str):
                    mask = leaf_mask(image_root / row["segmented_path"], size)
                    area = float(mask.mean())
                    focus = leaf_focus(m, mask)
                    stats.append({"path": row["path"], "correct": bool(row["correct"]),
                                  "leaf_area": area, "cam_on_leaf": focus,
                                  "focus_ratio": focus / area if area > 0 else np.nan})
    return np.stack(cams), pd.DataFrame(stats)


def plot_cam_grid(rows: pd.DataFrame, cams: np.ndarray, image_root: Path, size: int, title: str, save_path: Path):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.9 * n))
    axes = np.atleast_2d(axes)
    for i, ((_, row), cam) in enumerate(zip(rows.iterrows(), cams)):
        with Image.open(image_root / row["path"]) as im:
            rgb = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255
        plot_gradcam_triplet(rgb, cam, axes=axes[i])
        label = (f"true: {pretty_name(row['true']) if '___' in row['true'] else row['true']}  |  "
                 f"pred: {pretty_name(row['predicted']) if '___' in row['predicted'] else row['predicted']}"
                 f" ({row['confidence']:.2f})")
        for ax in axes[i]:
            ax.set_title("")
        axes[i, 0].set_title(label, fontsize=8, loc="left", color="#1a7f37" if row["correct"] else "#cf222e")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def write_markdown(path: Path, run: str, breakdown: dict, focus: dict, robustness: dict | None) -> None:
    lines = [f"# Failure analysis — `{run}` (test split)", "",
             "Generated by `python -m src.evaluation.failure_analysis`. Numbers only; the",
             "interpretation lives in `docs/results.md`.", "",
             "## Error overview", "",
             "| metric | value |", "|---|---|"]
    for key, value in breakdown.items():
        if isinstance(value, (int, float, str)) or value is None:
            lines.append(f"| {key} | {value if not isinstance(value, float) else f'{value:.4f}'} |")
    if "top_confused_pairs" in breakdown:
        lines += ["", "## Most frequent confusions", "", "| true | predicted | count |", "|---|---|---:|"]
        lines += [f"| {pretty_name(r['true'])} | {pretty_name(r['predicted'])} | {r['count']} |"
                  for r in breakdown["top_confused_pairs"]]
    lines += ["", "## Error rate by crop", "", "| crop | error rate |", "|---|---:|"]
    lines += [f"| {crop} | {rate:.2%} |" for crop, rate in breakdown["error_rate_by_crop"].items()]
    lines += ["", "## Where the model looks (Grad-CAM mass on the leaf)", "",
              "`cam_on_leaf` is the share of Grad-CAM mass inside the leaf mask taken from the",
              "segmented copy of the image; `leaf_area` is the share of the image the leaf covers.",
              "A ratio above 1 means attention is concentrated on the leaf.", "",
              "| subset | images | leaf_area | cam_on_leaf | focus_ratio |", "|---|---:|---:|---:|---:|"]
    for name, s in focus.items():
        lines.append(f"| {name} | {s['n']} | {s['leaf_area']:.3f} | {s['cam_on_leaf']:.3f} | {s['focus_ratio']:.2f} |")
    if robustness:
        variants = [v for v in ("color", "segmented", "segmented_gray") if v in robustness]
        lines += ["", "## Same test photos with the background changed", "",
                  "`segmented` is the dataset's background-removed copy (black outside the leaf);",
                  "`segmented_gray` puts the same leaf on a flat background in the colour of the lab",
                  "table, which keeps the image plausible and isolates the information the background",
                  "carried from the shock of an unseen black frame.", "",
                  "| metric | " + " | ".join(variants) + " |", "|---" * (len(variants) + 1) + "|"]
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = " | ".join(f"{robustness[v][key]:.4f}" for v in variants)
            lines.append(f"| {key} | {values} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--image-root", default="data/raw/plantvillage")
    parser.add_argument("--cam-sample", type=int, default=400, help="correct predictions used for the focus study")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    checkpoint = resolve_path(args.checkpoint)
    run = checkpoint.parent.name if checkpoint.is_file() else checkpoint.name
    image_root = resolve_path(args.image_root)
    report_dir = resolve_path("artifacts/reports") / run
    figure_dir = resolve_path("artifacts/figures") / run

    preds = pd.read_csv(report_dir / "test" / "predictions.csv")
    manifest = load_split(resolve_path(args.manifest), "test")[["path", "segmented_path"]]
    preds = preds.merge(manifest, on="path", how="left")
    model, meta = load_checkpoint(checkpoint, device)

    breakdown = error_breakdown(preds, meta["task"])

    gallery = pick_gallery_rows(preds, seed=args.seed)
    plot_prediction_gallery(gallery, image_root, title=f"{run} · test-set predictions (confidence-stratified)",
                            save_path=resolve_path("artifacts/sample_outputs") / run / "prediction_gallery.jpg")
    plt.close("all")

    wrong = preds[~preds["correct"]]
    correct = preds[preds["correct"]]
    # Random samples for the statistics; all errors if there are fewer than cam_sample.
    focus_rows = pd.concat([
        correct.sample(min(args.cam_sample, len(correct)), random_state=args.seed),
        wrong.sample(min(args.cam_sample, len(wrong)), random_state=args.seed),
    ])
    cams, stats = gradcam_study(model, meta, focus_rows, image_root, device)

    focus = {}
    for name, subset in (("correct", stats[stats["correct"]]), ("errors", stats[~stats["correct"]])):
        if len(subset):
            means = subset[["leaf_area", "cam_on_leaf", "focus_ratio"]].mean().round(4).to_dict()
            focus[name] = {"n": int(len(subset)), **means}

    size = meta["image_size"]
    plot_cam_grid(focus_rows.iloc[:6], cams[:6], image_root, size,
                  f"{run} · Grad-CAM on random correct test predictions", figure_dir / "gradcam_correct.jpg")
    top_errors = wrong.sort_values("confidence", ascending=False).head(6)
    if len(top_errors):
        error_cams, _ = gradcam_study(model, meta, top_errors, image_root, device)
        plot_cam_grid(top_errors, error_cams, image_root, size,
                      f"{run} · Grad-CAM on the most confident test errors", figure_dir / "gradcam_errors.jpg")

    robustness = {}
    keys = ("accuracy", "macro_f1", "balanced_accuracy")
    for variant, folder in (("color", "test"), ("segmented", "test_segmented"),
                            ("segmented_gray", "test_segmented_gray")):
        metrics_path = report_dir / folder / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            robustness[variant] = {k: metrics[k] for k in keys}
    if len(robustness) < 2:  # nothing to compare the colour result against
        robustness = None

    result = {"run": run, "errors": breakdown, "gradcam_leaf_focus": focus, "background_changed": robustness}
    with open(report_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    stats.to_csv(report_dir / "gradcam_leaf_focus.csv", index=False, float_format="%.4f")
    write_markdown(report_dir / "failure_analysis.md", run, breakdown, focus, robustness)
    print(f"{run}: {breakdown['n_errors']} errors / {breakdown['n_images']} test images; "
          f"Grad-CAM focus {focus}; report -> {display_path(report_dir / 'failure_analysis.md')}")


if __name__ == "__main__":
    main()
