"""Re-run the leakage audit on the committed manifest, without any images.

    python scripts/audit_manifest.py

Everything the audit needs (paths, groups, author leaf ids, md5) is in
data/splits/manifest.csv, so CI can check that nobody hand-edited a split into
a leaky state. Exits non-zero on failure.
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.grouping import add_identity_columns  # noqa: E402
from src.data.splits import audit_splits  # noqa: E402


def main():
    manifest = pd.read_csv(REPO / "data" / "splits" / "manifest.csv")
    with open(REPO / "data" / "splits" / "leakage_audit.json", encoding="utf-8") as f:
        buffer = json.load(f)["frame_buffer"]

    leaf_keys = manifest["leaf_key"]
    df = add_identity_columns(manifest.drop(columns=["leaf_key"]), leaf_map=None)
    df["leaf_key"] = leaf_keys.where(leaf_keys.notna(), None)

    report = audit_splits(df, buffer=buffer)
    print(report.summary())
    missing = {s: m for s, m in report.classes_missing.items() if m}
    if not report.passed or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
