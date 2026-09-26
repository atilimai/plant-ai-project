"""How much leakage does the camera-sequence rule leave behind?

We cannot measure that on the images that need the rule (they have no leaf ids),
so we measure it on the images that do: hide their leaf ids, split them with the
same contiguous-segment + frame-buffer rule, then use the hidden ids to count
held-out images whose leaf also appears in train. The results are written to
data/splits/grouping_calibration.csv and justify ``frame_buffer`` in configs/data.yaml.
"""

from __future__ import annotations

import pandas as pd

from src.data.splits import SPLITS, contiguous_sequence_splits, purge_frame_neighbours


def simulate_sequence_rule(identity: pd.DataFrame, buffer: int, ratios: dict[str, float], seed: int = 0) -> dict:
    """``identity`` is the output of ``grouping.add_identity_columns``."""
    known = identity[identity["leaf_key"].notna() & identity["frame"].notna()].reset_index(drop=True)
    truth = known["leaf_key"]
    hidden = known.assign(leaf_key=None)
    # Camera-frame duplicates are still visible without leaf ids, everything else is not.
    hidden["group_id"] = hidden["class_name"] + "/" + hidden["identifier"]

    hidden["split"] = contiguous_sequence_splits(hidden, ratios, seed=seed)
    held_before = hidden["split"].isin(["val", "test"]).sum()
    hidden["split"] = purge_frame_neighbours(hidden, buffer)

    in_split = {s: set(truth[hidden["split"] == s]) for s in SPLITS}
    held_out = hidden["split"].isin(["val", "test"])
    return {
        "frame_buffer": buffer,
        "heldout_purged_share": round(1 - held_out.sum() / held_before, 4),
        "heldout_sharing_leaf_with_train": round(truth[held_out].isin(in_split["train"]).mean(), 4),
        "test_sharing_leaf_with_val": round(truth[hidden["split"] == "test"].isin(in_split["val"]).mean(), 4),
        "n_images": int(len(known)),
    }


def calibration_table(identity: pd.DataFrame, ratios: dict[str, float], buffers, seed: int = 0) -> pd.DataFrame:
    return pd.DataFrame([simulate_sequence_rule(identity, b, ratios, seed) for b in buffers])
