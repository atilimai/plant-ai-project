import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.labels import PLANTVILLAGE_CLASSES

CLASSES = ["Apple___Apple_scab", "Apple___healthy", "Tomato___Early_blight"]


@pytest.fixture
def image_tree(tmp_path):
    """A tiny PlantVillage-shaped folder: 3 classes x 12 images, colour + segmented."""
    rng = np.random.default_rng(0)
    rows = []
    for c_idx, class_name in enumerate(CLASSES):
        for i in range(12):
            name = f"0000000{c_idx}-0000-0000-0000-{i:012d}___CAM {1000 + i}"
            color = tmp_path / "color" / class_name / f"{name}.JPG"
            seg = tmp_path / "segmented" / class_name / f"{name}_final_masked.jpg"
            color.parent.mkdir(parents=True, exist_ok=True)
            seg.parent.mkdir(parents=True, exist_ok=True)
            pixels = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
            Image.fromarray(pixels).save(color, format="JPEG")
            Image.fromarray(pixels // 2).save(seg, format="JPEG")
            rows.append({
                "path": f"color/{class_name}/{name}.JPG",
                "class_name": class_name,
                "split": ["train", "train", "val", "test"][i % 4],
                "has_segmented": True,
            })
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return tmp_path, manifest_path


@pytest.fixture
def all_classes():
    return list(PLANTVILLAGE_CLASSES)
