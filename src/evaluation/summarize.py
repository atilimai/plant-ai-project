"""Collect metrics.json files from every run into one comparison table.

    python -m src.evaluation.summarize

Writes artifacts/reports/summary.json and artifacts/reports/summary.md.
"""

from __future__ import annotations

import json

import pandas as pd

from src.utils.config import resolve_path

COLUMNS = {
    "multiclass": ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "ece"],
    "binary": ["accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "roc_auc", "ece"],
}


def load_metrics(reports_root=None) -> pd.DataFrame:
    root = resolve_path(reports_root or "artifacts/reports")
    rows = []
    for path in sorted(root.glob("*/*/metrics.json")):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        row = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
        if "derived_binary" in m:
            for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc"):
                row[f"derived_binary_{k}"] = m["derived_binary"].get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| run | split | " + " | ".join(columns) + " |"
    lines = [header, "|" + "---|" * (len(columns) + 2)]
    for _, r in df.iterrows():
        split = r["split"] if r["variant"] == "color" else f"{r['split']} ({r['variant']})"
        values = " | ".join("–" if pd.isna(r.get(c)) else f"{r[c]:.4f}" for c in columns)
        lines.append(f"| {r['run']} | {split} | {values} |")
    return "\n".join(lines)


def summarize_runs(reports_root=None) -> pd.DataFrame:
    df = load_metrics(reports_root)
    if df.empty:
        print("No metrics.json found under artifacts/reports")
        return df
    df = df.sort_values(["task", "run", "split", "variant"])
    out = resolve_path(reports_root or "artifacts/reports")
    df.to_json(out / "summary.json", orient="records", indent=2)

    parts = ["# Results summary", "",
             "All numbers come from `metrics.json` files written by `src.evaluation.evaluate`.",
             "Test is the leaf-level held-out split (see `data/splits/README.md`). The two extra rows",
             "per run are the same test photos with the background changed: `segmented` is the",
             "dataset's background-removed copy (black), `segmented_gray` puts the leaf on a flat",
             "background in the colour of the lab table. See `docs/results.md`.", ""]
    for task in ("multiclass", "binary"):
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        parts += [f"## {task.capitalize()}", "", _table(sub, COLUMNS[task]), ""]
        if task == "multiclass" and "derived_binary_accuracy" in sub:
            cols = ["derived_binary_accuracy", "derived_binary_balanced_accuracy",
                    "derived_binary_sensitivity", "derived_binary_specificity", "derived_binary_roc_auc"]
            parts += ["Healthy-vs-diseased read off the multiclass models:", "", _table(sub, cols), ""]
    (out / "summary.md").write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))
    return df


if __name__ == "__main__":
    summarize_runs()
