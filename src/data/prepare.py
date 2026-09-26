"""Download PlantVillage, rebuild leaf groups, split, audit and write the manifest.

    python -m src.data.prepare --config configs/data.yaml

Steps whose output already exists are skipped, so re-running is cheap.
Committed outputs in data/splits/: manifest.csv, split_summary.json,
leakage_audit.json, grouping_calibration.csv. Images stay under data/raw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.data.calibration import calibration_table
from src.data.grouping import add_identity_columns, assign_leaf_groups, load_leaf_map
from src.data.labels import PLANTVILLAGE_CLASSES, is_healthy
from src.data.splits import (
    EXCLUDED,
    SPLITS,
    assign_splits,
    audit_splits,
    purge_frame_neighbours,
    sequence_split_mask,
)
from src.utils.config import display_path, load_config, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_COLUMNS = [
    "path", "class_name", "split", "split_rule", "group_id", "group_source", "leaf_key", "md5", "has_segmented",
]


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None) -> Path:
    if dest.exists():
        if expected_sha256 is None or sha256sum(dest) == expected_sha256:
            return dest
        print(f"{dest.name} exists but its checksum does not match, downloading again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while block := response.read(1 << 20):
                out.write(block)
                bar.update(len(block))
    if expected_sha256 and (actual := sha256sum(tmp)) != expected_sha256:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {dest.name}: expected {expected_sha256}, got {actual}")
    tmp.replace(dest)
    return dest


def extract_variants(zip_path: Path, image_root: Path, variants: list[str]) -> None:
    todo = [v for v in variants if not (image_root / v).is_dir()]
    if not todo:
        return
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.infolist()
            if not m.is_dir() and any(m.filename.startswith(f"raw/{v}/") for v in todo)
        ]
        for m in tqdm(members, desc=f"extracting {', '.join(todo)}"):
            target = image_root / m.filename.removeprefix("raw/")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def segmented_name(color_rel_path: str) -> str:
    """color/<class>/<stem>.JPG -> segmented/<class>/<stem>_final_masked.jpg"""
    _, class_name, file_name = color_rel_path.split("/")
    return f"segmented/{class_name}/{Path(file_name).stem}_final_masked.jpg"


def scan_images(image_root: Path) -> pd.DataFrame:
    rows = []
    for class_name in tqdm(PLANTVILLAGE_CLASSES, desc="hashing classes"):
        class_dir = image_root / "color" / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder {class_dir}")
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = f"color/{class_name}/{f.name}"
            rows.append({
                "path": rel,
                "class_name": class_name,
                "md5": hashlib.md5(f.read_bytes()).hexdigest(),
                "has_segmented": (image_root / segmented_name(rel)).exists(),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    splits = [*SPLITS, EXCLUDED]
    per_class = (
        df.pivot_table(index="class_name", columns="split", values="path", aggfunc="count", fill_value=0)
        .reindex(columns=splits, fill_value=0)
        .reindex(PLANTVILLAGE_CLASSES)
    )
    kept = df[df["split"].isin(SPLITS)]
    return {
        "n_images": int(len(df)),
        "n_classes": int(df["class_name"].nunique()),
        "n_groups": int(df["group_id"].nunique()),
        "images_per_split": {s: int((df["split"] == s).sum()) for s in splits},
        "groups_per_split": {s: int(kept.loc[kept["split"] == s, "group_id"].nunique()) for s in SPLITS},
        "healthy_share_per_split": {
            s: round(float(kept.loc[kept["split"] == s, "class_name"].map(is_healthy).mean()), 4) for s in SPLITS
        },
        "group_source_counts": {k: int(v) for k, v in df["group_source"].value_counts().items()},
        "images_with_author_leaf_id": int(df["leaf_key"].notna().sum()),
        "images_per_class": {c: {s: int(v) for s, v in row.items()} for c, row in per_class.iterrows()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--skip-download", action="store_true", help="images are already extracted")
    parser.add_argument("--rescan", action="store_true", help="ignore the cached file scan")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    src, paths = cfg["source"], cfg["paths"]
    raw_dir, image_root = resolve_path(paths["raw_dir"]), resolve_path(paths["image_root"])
    interim_dir, splits_dir = resolve_path(paths["interim_dir"]), resolve_path(paths["splits_dir"])
    base_url = f"https://huggingface.co/datasets/{src['hf_repo']}/resolve/{src['revision']}"

    leaf_map_path = download(f"{base_url}/{src['leaf_map_file']}", raw_dir / "leaf-map.json", src["leaf_map_sha256"])
    if not args.skip_download:
        zip_path = download(f"{base_url}/{src['zip_file']}", raw_dir / "plantvillage_data.zip", src["zip_sha256"])
        extract_variants(zip_path, image_root, src["variants"])

    scan_cache = interim_dir / "scan.csv"
    if scan_cache.exists() and not args.rescan:
        scanned = pd.read_csv(scan_cache)
    else:
        scanned = scan_images(image_root)
        interim_dir.mkdir(parents=True, exist_ok=True)
        scanned.to_csv(scan_cache, index=False)
    print(f"{len(scanned)} colour images, {int(scanned['has_segmented'].sum())} with a segmented copy")

    leaf_map = load_leaf_map(leaf_map_path)
    group_cfg, split_cfg = cfg["grouping"], cfg["split"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    calib = calibration_table(
        add_identity_columns(scanned, leaf_map), split_cfg["ratios"],
        buffers=group_cfg["calibration_buffers"], seed=split_cfg["seed"],
    )
    calib.to_csv(splits_dir / "grouping_calibration.csv", index=False)
    print("Sequence rule simulated on images with author leaf ids:\n" + calib.to_string(index=False))

    df = assign_leaf_groups(scanned, leaf_map)
    df["split_rule"] = sequence_split_mask(df).map({True: "camera_sequence", False: "leaf_group"})
    df["split"] = assign_splits(df, split_cfg["ratios"], seed=split_cfg["seed"])
    df["split"] = purge_frame_neighbours(df, group_cfg["frame_buffer"])
    report = audit_splits(df, buffer=group_cfg["frame_buffer"], strict=True)
    print(report.summary())
    print(f"{int((df['split'] == EXCLUDED).sum())} held-out images excluded as frame neighbours of another split")

    summary = summarize(df)
    audit = report.to_dict()
    audit["frame_buffer"] = group_cfg["frame_buffer"]
    with open(splits_dir / "leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    df[MANIFEST_COLUMNS].to_csv(splits_dir / "manifest.csv", index=False)
    print(f"Wrote {display_path(splits_dir / 'manifest.csv')}: {summary['images_per_split']}")


if __name__ == "__main__":
    main()
