import pandas as pd
import pytest

from src.data.grouping import (
    DisjointSet,
    assign_leaf_groups,
    camera_identifier,
    lookup_leaf,
    named_leaf,
    split_frame_number,
)


@pytest.mark.parametrize("file_name, expected", [
    ("47f24ceb-34e8-4c13-acac-e48549bc7cfa___RS_NLB 3932 copy 2.jpg", "rs_nlb 3932"),
    ("ef5c7a36-4a79-419f-94e9-ad1120a68dad___RS_NLB 3932 copy.jpg", "rs_nlb 3932"),
    ("00a14441-7a62-4034-bc40-b196aeab2785___RS_NLB 3932.JPG", "rs_nlb 3932"),
    ("e51367c1-d582-45e3-acf6-b0e652b43d3a___RS_HL 1873_final_masked.jpg", "rs_hl 1873"),
    ("RS_Rust 1563.JPG", "rs_rust 1563"),  # a few Corn files have no uuid prefix
])
def test_camera_identifier(file_name, expected):
    assert camera_identifier(file_name) == expected


def test_frame_number_ignores_duplicate_export_suffix():
    assert split_frame_number("rs_hl 5529 1") == ("rs_hl", 5529)
    assert split_frame_number("ghlb2 leaf 8664") == ("ghlb2 leaf", 8664)
    assert split_frame_number("screen shot at pm") == ("screen shot at pm", None)


def test_named_leaf_collapses_days():
    assert named_leaf("ghlb_ps leaf 23.7 day 13") == "ghlb_ps leaf 23.7"
    assert named_leaf("ghlb leaf 2.1 day 16") == "ghlb leaf 2.1"
    assert named_leaf("ghlb2 leaf 8664") is None


def test_lookup_leaf_prefers_same_class_then_same_crop():
    leaf_map = {
        "cam 1": ["Apple___Apple_scab:::3.0"],
        "cam 2": ["Apple_Frogeye Spot:::7.0"],
        "cam 3": ["Tomato___Early_blight:::1.0", "Apple___healthy:::9.0"],
        "cam 4": ["Grape___healthy:::2.0"],
    }
    assert lookup_leaf("cam 1", "Apple___Apple_scab", leaf_map) == "Apple___Apple_scab:::3.0"
    assert lookup_leaf("cam 2", "Apple___Black_rot", leaf_map) == "Apple_Frogeye Spot:::7.0"
    assert lookup_leaf("cam 3", "Apple___healthy", leaf_map) == "Apple___healthy:::9.0"
    assert lookup_leaf("cam 4", "Apple___healthy", leaf_map) is None  # different crop
    assert lookup_leaf("missing", "Apple___healthy", leaf_map) is None


def test_disjoint_set_merges_transitively():
    dsu = DisjointSet(5)
    dsu.union(0, 1)
    dsu.union(3, 4)
    dsu.union(1, 4)
    assert len({dsu.find(i) for i in (0, 1, 3, 4)}) == 1
    assert dsu.find(2) == 2


def test_assign_leaf_groups_combines_all_evidence():
    uuid = "00000000-0000-0000-0000-00000000000{}___"
    df = pd.DataFrame({
        "class_name": ["Apple___Apple_scab"] * 4 + ["Tomato___Late_blight"] * 3 + ["Apple___healthy"],
        "path": [
            f"color/Apple___Apple_scab/{uuid.format(0)}CAM 10.JPG",
            f"color/Apple___Apple_scab/{uuid.format(1)}CAM 11.JPG",     # same leaf via leaf map
            f"color/Apple___Apple_scab/{uuid.format(2)}CAM 12 copy.jpg",
            f"color/Apple___Apple_scab/{uuid.format(3)}CAM 12.JPG",     # same frame as the copy
            f"color/Tomato___Late_blight/{uuid.format(4)}GHLB Leaf 5 Day 1.jpg",
            f"color/Tomato___Late_blight/{uuid.format(5)}GHLB Leaf 5 Day 9.jpg",  # same named leaf
            f"color/Tomato___Late_blight/{uuid.format(6)}GHLB Leaf 6 Day 1.jpg",
            f"color/Apple___healthy/{uuid.format(7)}OTHER 1.JPG",
        ],
        "md5": ["a", "b", "c", "d", "e", "f", "g", "a"],  # last row duplicates the first file
    })
    leaf_map = {"cam 10": ["Apple___Apple_scab:::1.0"], "cam 11": ["Apple___Apple_scab:::1.0"]}
    out = assign_leaf_groups(df, leaf_map)
    g = out["group_id"].tolist()

    assert g[0] == g[1] == g[7]
    assert g[2] == g[3] != g[0]
    assert g[4] == g[5] != g[6]
    assert out.loc[0, "group_source"] == "leaf_map"
    assert out.loc[2, "group_source"] == "camera_frame"
    assert out.loc[4, "group_source"] == "named_leaf"
    assert out.loc[6, "group_source"] == "singleton"


def test_group_ids_do_not_depend_on_row_order():
    df = pd.DataFrame({
        "class_name": ["Apple___healthy"] * 3,
        "path": ["color/Apple___healthy/A 1.JPG", "color/Apple___healthy/B 1.JPG", "color/Apple___healthy/A 1 copy.jpg"],
    })
    forward = assign_leaf_groups(df).set_index("path")["group_id"]
    backward = assign_leaf_groups(df.iloc[::-1]).set_index("path")["group_id"]
    assert forward.sort_index().equals(backward.sort_index())
