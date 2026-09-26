"""Train/val/test assignment at the leaf-group level, plus the leakage audit."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLITS = ("train", "val", "test")
EXCLUDED = "excluded"
# When two neighbouring frames end up in different splits, the lower-priority
# image is dropped: we never touch train, and val gives way to test.
_PRIORITY = {"train": 0, "test": 1, "val": 2}


class LeakageError(RuntimeError):
    pass


def make_group_splits(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int = 42,
    group_col: str = "group_id",
    stratify_col: str = "class_name",
    n_folds: int = 20,
) -> pd.Series:
    """Assign every row to train/val/test without splitting a group.

    We cut the data into ``n_folds`` stratified group folds and hand out whole folds,
    so ratios are honoured in steps of 1/n_folds (5% by default). That is coarse,
    but StratifiedGroupKFold keeps per-class proportions far better than shuffling
    groups directly, which matters for the rare classes (~150 images).
    """
    total = sum(ratios.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {total}")

    fold_counts = {name: int(round(ratios.get(name, 0.0) * n_folds)) for name in SPLITS}
    fold_counts["train"] = n_folds - fold_counts["val"] - fold_counts["test"]
    if min(fold_counts.values()) <= 0:
        raise ValueError(f"Ratios {ratios} leave an empty split with n_folds={n_folds}")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.empty(len(df), dtype=int)
    with warnings.catch_warnings():
        # A couple of classes have only a handful of id-less singletons in this
        # branch; sklearn warns about it, but those rows are still assigned.
        warnings.filterwarnings("ignore", message="The least populated class")
        for fold, (_, idx) in enumerate(sgkf.split(df, df[stratify_col], df[group_col])):
            fold_of[idx] = fold

    # Fold contents are already shuffled (shuffle=True), so a fixed layout is fine.
    names = (["test"] * fold_counts["test"] + ["val"] * fold_counts["val"]
             + ["train"] * fold_counts["train"])
    return pd.Series([names[f] for f in fold_of], index=df.index, name="split")


def sequence_split_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose whole group has no author leaf id but does have camera frame numbers."""
    no_leaf_id = df["leaf_key"].isna().groupby(df["group_id"]).transform("all")
    has_frame = df["frame"].notna().groupby(df["group_id"]).transform("any")
    return no_leaf_id & has_frame


def contiguous_sequence_splits(df: pd.DataFrame, ratios: dict[str, float], seed: int = 42) -> pd.Series:
    """Split each camera sequence into contiguous test / val / train segments.

    Without leaf ids, the only thing we know is that neighbouring frames are often
    the same leaf. Random assignment would create a split boundary every few frames;
    contiguous segments create at most three per sequence (train|test, test|val,
    val|train), and ``purge_frame_neighbours`` then drops a small buffer at each.
    The segment layout is rotated by a random offset per sequence, so held-out
    segments are not always the first frames of a session.

    Groups stay intact: a group is placed by its first frame and takes all its rows.
    """
    rng = np.random.default_rng(seed)
    units = df.groupby("group_id").agg(
        sequence=("sequence", "first"), frame=("frame", "min"), size=("path", "size")
    )
    cut_test = ratios["test"]
    cut_val = ratios["test"] + ratios["val"]

    assigned = {}
    for _, seq_units in units.groupby("sequence", sort=True):
        seq_units = seq_units.sort_values("frame", kind="stable")
        sizes = seq_units["size"].to_numpy()
        midpoint = (np.cumsum(sizes) - sizes / 2) / sizes.sum()
        position = (midpoint + rng.random()) % 1.0
        names = np.where(position < cut_test, "test", np.where(position < cut_val, "val", "train"))
        assigned.update(zip(seq_units.index, names))
    return df["group_id"].map(assigned).rename("split")


def assign_splits(df: pd.DataFrame, ratios: dict[str, float], seed: int = 42, n_folds: int = 20) -> pd.Series:
    """Leaf groups go through StratifiedGroupKFold, id-less camera sequences are cut contiguously."""
    by_sequence = sequence_split_mask(df)
    split = pd.Series(index=df.index, dtype=object, name="split")
    split[~by_sequence] = make_group_splits(df[~by_sequence], ratios, seed=seed, n_folds=n_folds)
    if by_sequence.any():
        split[by_sequence] = contiguous_sequence_splits(df[by_sequence], ratios, seed=seed)
    return split


def frame_conflicts(df: pd.DataFrame, buffer: int, split_col: str = "split") -> np.ndarray:
    """Boolean mask of rows that sit within ``buffer`` frames of a higher-priority split.

    Only pairs where at least one image lacks an author leaf id count: if both have
    one, the leaf map already tells us whether they are the same leaf.
    """
    conflict = np.zeros(len(df), dtype=bool)
    if buffer <= 0:
        return conflict

    usable = df["frame"].notna() & df[split_col].isin(_PRIORITY)
    for _, sub in df[usable].groupby("sequence", sort=False):
        if sub["leaf_key"].notna().all():
            continue
        sub = sub.sort_values("frame")
        frames = sub["frame"].to_numpy(dtype=np.int64)
        priority = sub[split_col].map(_PRIORITY).to_numpy()
        unknown = sub["leaf_key"].isna().to_numpy()
        positions = sub.index.to_numpy()
        lo = np.searchsorted(frames, frames - buffer, side="left")
        hi = np.searchsorted(frames, frames + buffer, side="right")
        for k in np.flatnonzero(priority > 0):
            window = slice(lo[k], hi[k])
            higher = priority[window] < priority[k]
            if unknown[k]:
                hit = higher.any()
            else:
                hit = (higher & unknown[window]).any()
            if hit:
                conflict[df.index.get_loc(positions[k])] = True
    return conflict


def purge_frame_neighbours(df: pd.DataFrame, buffer: int, split_col: str = "split") -> pd.Series:
    """Mark held-out images that are frame-neighbours of another split as 'excluded'.

    One pass is enough: excluding an image never creates a new conflict, because the
    image it conflicted with has higher priority and stays where it is.
    """
    split = df[split_col].copy()
    split[frame_conflicts(df, buffer, split_col)] = EXCLUDED
    return split


@dataclass
class AuditReport:
    n_images: dict[str, int]
    n_groups: dict[str, int]
    overlaps: dict[str, dict[str, int]] = field(default_factory=dict)
    frame_buffer_violations: int | None = None
    classes_missing: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        shared = any(v for per_key in self.overlaps.values() for v in per_key.values())
        return not shared and not self.frame_buffer_violations

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_images": self.n_images,
            "n_groups": self.n_groups,
            "overlaps": self.overlaps,
            "frame_buffer_violations": self.frame_buffer_violations,
            "classes_missing": self.classes_missing,
        }

    def summary(self) -> str:
        lines = [f"Leakage audit: {'PASSED' if self.passed else 'FAILED'}"]
        for split, n in self.n_images.items():
            lines.append(f"  {split:8s} images={n:6d} groups={self.n_groups[split]:6d}")
        for key, per_pair in self.overlaps.items():
            shown = ", ".join(f"{pair}={n}" for pair, n in per_pair.items())
            lines.append(f"  shared {key}: {shown}")
        if self.frame_buffer_violations is not None:
            lines.append(f"  frame-buffer violations: {self.frame_buffer_violations}")
        for split, missing in self.classes_missing.items():
            if missing:
                lines.append(f"  {split} is missing classes: {missing}")
        return "\n".join(lines)


def audit_splits(
    df: pd.DataFrame,
    keys: tuple[str, ...] = ("group_id", "leaf_key", "identifier_in_class", "md5"),
    buffer: int | None = None,
    strict: bool = False,
) -> AuditReport:
    """Check that nothing identifying a leaf or a photo is shared across splits.

    ``identifier_in_class`` is derived from ``class_name`` + ``identifier``. Keys whose
    columns are missing are skipped, so the audit also runs on a bare manifest.
    Excluded rows are ignored. With ``buffer`` set, the frame-neighbour rule used by
    ``purge_frame_neighbours`` is re-checked from scratch as well.
    """
    df = df[df["split"].isin(SPLITS)].copy()
    if "identifier" in df and "class_name" in df:
        df["identifier_in_class"] = df["class_name"] + "/" + df["identifier"]

    present = [s for s in SPLITS if (df["split"] == s).any()]
    report = AuditReport(
        n_images={s: int((df["split"] == s).sum()) for s in present},
        n_groups={s: int(df.loc[df["split"] == s, "group_id"].nunique()) for s in present},
    )

    for key in keys:
        if key not in df:
            continue
        values = {s: set(df.loc[df["split"] == s, key].dropna()) for s in present}
        report.overlaps[key] = {
            f"{a}-{b}": len(values[a] & values[b])
            for i, a in enumerate(present) for b in present[i + 1:]
        }

    if buffer is not None and {"sequence", "frame", "leaf_key"} <= set(df.columns):
        report.frame_buffer_violations = int(frame_conflicts(df.reset_index(drop=True), buffer).sum())

    if "class_name" in df:
        all_classes = set(df["class_name"])
        report.classes_missing = {
            s: sorted(all_classes - set(df.loc[df["split"] == s, "class_name"])) for s in present
        }

    if strict and not report.passed:
        raise LeakageError(report.summary())
    return report
