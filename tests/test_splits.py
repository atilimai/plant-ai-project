import numpy as np
import pandas as pd
import pytest

from src.data.grouping import add_identity_columns
from src.data.splits import (
    EXCLUDED,
    LeakageError,
    assign_splits,
    audit_splits,
    contiguous_sequence_splits,
    make_group_splits,
    purge_frame_neighbours,
)

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def leaf_frame(n_classes=4, leaves_per_class=60, images_per_leaf=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        for leaf in range(leaves_per_class):
            for k in range(rng.integers(1, images_per_leaf + 1)):
                rows.append({
                    "path": f"color/class{c}/leaf{leaf}_{k}.jpg",
                    "class_name": f"class{c}",
                    "group_id": f"c{c}-leaf{leaf}",
                })
    return pd.DataFrame(rows)


def sequence_frame(n=300, leaf_key_every=None):
    df = pd.DataFrame({
        "path": [f"color/Corn___rust/CAM {1000 + i}.JPG" for i in range(n)],
        "class_name": "Corn___rust",
    })
    df = add_identity_columns(df, leaf_map=None)
    df["group_id"] = df["path"]
    if leaf_key_every:
        df.loc[::leaf_key_every, "leaf_key"] = "known"
    return df


def test_group_split_never_breaks_a_group_and_keeps_ratios():
    df = leaf_frame()
    df["split"] = make_group_splits(df, RATIOS, seed=1)
    assert df.groupby("group_id")["split"].nunique().max() == 1
    share = df["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.05)
    assert share["val"] == pytest.approx(0.15, abs=0.05)
    assert share["test"] == pytest.approx(0.15, abs=0.05)
    assert (df.groupby("class_name")["split"].nunique() == 3).all()


def test_group_split_is_deterministic():
    df = leaf_frame()
    assert make_group_splits(df, RATIOS, seed=7).equals(make_group_splits(df, RATIOS, seed=7))


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        make_group_splits(leaf_frame(), {"train": 0.8, "val": 0.15, "test": 0.15})


def test_contiguous_sequence_split_makes_three_segments():
    df = sequence_frame()
    split = contiguous_sequence_splits(df, RATIOS, seed=3)
    changes = (split != split.shift()).sum() - 1
    assert changes <= 3
    assert split.value_counts(normalize=True)["test"] == pytest.approx(0.15, abs=0.01)


def test_purge_removes_held_out_frames_near_train():
    df = sequence_frame()
    df["split"] = contiguous_sequence_splits(df, RATIOS, seed=3)
    before = df["split"].copy()
    df["split"] = purge_frame_neighbours(df, buffer=5)

    assert (df["split"] == "train").equals(before == "train"), "train must never be purged"
    assert (df["split"] == EXCLUDED).sum() > 0
    kept = df[df["split"] != EXCLUDED].sort_values("frame")
    train_frames = kept.loc[kept["split"] == "train", "frame"].to_numpy(dtype=int)
    for frame in kept.loc[kept["split"] != "train", "frame"].to_numpy(dtype=int):
        assert np.abs(train_frames - frame).min() > 5
    assert audit_splits(df, buffer=5).frame_buffer_violations == 0


def test_purge_ignores_pairs_that_both_have_leaf_ids():
    df = sequence_frame(n=4)
    df["leaf_key"] = ["a", "b", "c", "d"]
    df["split"] = ["train", "test", "train", "val"]
    assert (purge_frame_neighbours(df, buffer=3) == df["split"]).all()


def test_assign_splits_routes_id_less_sequences_separately():
    seq = sequence_frame(n=200)
    leaves = leaf_frame(n_classes=1, leaves_per_class=80)
    leaves["leaf_key"] = leaves["group_id"]
    leaves["frame"] = pd.array([None] * len(leaves), dtype="Int64")
    leaves["sequence"] = "other"
    df = pd.concat([seq, leaves], ignore_index=True)
    split = assign_splits(df, RATIOS, seed=0)
    seq_split = split.iloc[: len(seq)]
    assert (seq_split != seq_split.shift()).sum() - 1 <= 3


def test_audit_catches_shared_groups():
    df = leaf_frame()
    df["split"] = make_group_splits(df, RATIOS)
    assert audit_splits(df).passed

    leaky = df.copy()
    victim = leaky.index[leaky["split"] == "test"][0]
    leaky.loc[victim, "group_id"] = leaky.loc[leaky["split"] == "train", "group_id"].iloc[0]
    report = audit_splits(leaky)
    assert not report.passed
    assert report.overlaps["group_id"]["train-test"] == 1
    with pytest.raises(LeakageError):
        audit_splits(leaky, strict=True)


def test_audit_ignores_excluded_rows():
    df = leaf_frame()
    df["split"] = make_group_splits(df, RATIOS)
    train_group = df.loc[df["split"] == "train", "group_id"].iloc[0]
    extra = pd.DataFrame([{"path": "x.jpg", "class_name": "class0", "group_id": train_group, "split": EXCLUDED}])
    assert audit_splits(pd.concat([df, extra], ignore_index=True)).passed
