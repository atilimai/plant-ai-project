"""Leaf-level grouping for PlantVillage.

PlantVillage has several photos of every physical leaf, but no clean leaf id.
We rebuild one from the evidence we have and merge it with union-find, so any two
images connected by *some* evidence land in the same group:

1. ``leaf-map.json`` from the dataset authors: camera file name -> leaf number,
   for ~41k of the 54k colour images.
2. The camera file name. Some photos were cut into several tiles
   ("RS_NLB 3932.JPG", "RS_NLB 3932 copy.jpg", "RS_NLB 3932 copy 2.jpg").
3. Leaf numbers written into the name. The Tomato late blight time series is
   named "GHLB_PS Leaf 23.7 Day 13": one leaf photographed on different days.
4. Byte-identical files.

Images the leaf map does not cover (all Corn and Squash classes, Tomato Target
Spot and mosaic virus, Grape healthy, part of several Tomato classes) often end
up as singletons here. They are not split like the rest: see
``splits.contiguous_sequence_splits``.

Over-merging is the safe failure mode: a group that is too big costs a little
split flexibility, a group that is too small leaks into the test set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_UUID_PREFIX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}___", re.I)
_COPY_SUFFIX = re.compile(r"\s+copy(\s+\d+)?$", re.I)
# "rs_hl 5529 1" is frame 5529 exported twice; the short trailing number is not the frame.
_FRAME = re.compile(r"^(?P<prefix>.*?)[\s_]*(?P<number>\d+)(?:\s+\d{1,2})?$")
_LEAF_DAY = re.compile(r"^(?P<leaf>.*\bleaf\s+\d+(?:\.\d+)?)\.?\s+day\s+[\d.]+$")


def camera_identifier(file_name: str) -> str:
    """Recover the original camera file name, lower-cased, without extension.

    >>> camera_identifier("47f24ceb-34e8-4c13-acac-e48549bc7cfa___RS_NLB 3932 copy 2.jpg")
    'rs_nlb 3932'
    """
    stem = Path(file_name).stem
    stem = stem.removesuffix("_final_masked")
    stem = _UUID_PREFIX.sub("", stem)
    stem = _COPY_SUFFIX.sub("", stem.strip())
    return stem.strip().lower()


def named_leaf(identifier: str) -> str | None:
    """'ghlb_ps leaf 23.7 day 13' -> 'ghlb_ps leaf 23.7'; None for ordinary frame names."""
    m = _LEAF_DAY.match(identifier)
    return m.group("leaf") if m else None


def split_frame_number(identifier: str) -> tuple[str, int | None]:
    """'rs_nlb 3932' -> ('rs_nlb', 3932); 'rs_hl 5529 1' -> ('rs_hl', 5529)."""
    m = _FRAME.match(identifier)
    if not m:
        return identifier, None
    return m.group("prefix").strip(), int(m.group("number"))


def load_leaf_map(path: str | Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lookup_leaf(identifier: str, class_name: str, leaf_map: dict[str, list[str]]) -> str | None:
    """Return the author-provided leaf key (e.g. 'Peach___healthy:::54.0') or None.

    A few camera names occur in more than one spreadsheet; we only accept an entry
    whose sheet belongs to the same crop. Sheet names are not always the class
    folder name ("Apple_Frogeye Spot" holds the Apple___Black_rot leaves), so the
    crop is the most specific thing we can match on reliably.
    """
    candidates = leaf_map.get(identifier)
    if not candidates:
        return None

    exact = [c for c in candidates if c.split(":::")[0] == class_name]
    if len(exact) == 1:
        return exact[0]

    crop = _crop_token(class_name)
    same_crop = [c for c in candidates if _crop_token(c.split(":::")[0]) == crop]
    if len(same_crop) == 1:
        return same_crop[0]
    return None


def _crop_token(name: str) -> str:
    return re.split(r"[_\s(,]", name, maxsplit=1)[0].lower()


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def add_identity_columns(df: pd.DataFrame, leaf_map: dict[str, list[str]] | None) -> pd.DataFrame:
    """Add ``identifier``, ``sequence``, ``frame`` and ``leaf_key``."""
    df = df.reset_index(drop=True).copy()
    df["identifier"] = [camera_identifier(Path(p).name) for p in df["path"]]
    parsed = [split_frame_number(i) for i in df["identifier"]]
    df["sequence"] = df["class_name"] + "/" + pd.Series([p for p, _ in parsed])
    df["frame"] = pd.array([n for _, n in parsed], dtype="Int64")
    if leaf_map is None:
        df["leaf_key"] = None
    else:
        df["leaf_key"] = [lookup_leaf(i, c, leaf_map) for i, c in zip(df["identifier"], df["class_name"])]
    return df


def assign_leaf_groups(
    df: pd.DataFrame,
    leaf_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with identity columns plus ``group_id`` and ``group_source``.

    ``df`` needs ``path`` and ``class_name``; an ``md5`` column is used if present.
    ``group_source`` records the first kind of evidence that grouped an image with
    another one ("singleton" if nothing did).
    """
    df = add_identity_columns(df, leaf_map)
    dsu = DisjointSet(len(df))
    source = np.array(["singleton"] * len(df), dtype=object)

    def link_by(keys: pd.Series, label: str) -> None:
        for members in keys.dropna().groupby(keys.dropna()).indices.values():
            if len(members) < 2:
                continue
            positions = keys.dropna().index.to_numpy()[members]
            for other in positions[1:]:
                dsu.union(int(positions[0]), int(other))
            fresh = source[positions] == "singleton"
            source[positions[fresh]] = label

    link_by(df["leaf_key"], "leaf_map")
    link_by(df["class_name"] + "/" + df["identifier"], "camera_frame")
    named = pd.Series([named_leaf(i) for i in df["identifier"]], index=df.index)
    link_by(df["class_name"] + "/" + named, "named_leaf")
    if "md5" in df:
        link_by(df["md5"], "exact_duplicate")

    roots = np.array([dsu.find(i) for i in range(len(df))])
    # Number groups in path order so ids are stable across runs and machines.
    order = np.argsort(df["path"].to_numpy(), kind="stable")
    group_number = {}
    for i in order:
        group_number.setdefault(roots[i], len(group_number))
    df["group_id"] = [f"leaf{group_number[r]:05d}" for r in roots]
    df["group_source"] = source
    return df
