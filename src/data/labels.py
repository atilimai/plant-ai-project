"""Class vocabulary for PlantVillage and the two label views we train on."""

from __future__ import annotations

# Folder names exactly as they appear in the dataset. Order matters: it defines
# the multiclass label index and must never be reshuffled once models exist.
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

BINARY_CLASSES = ["healthy", "diseased"]
TASKS = ("binary", "multiclass")


def split_class_name(class_name: str) -> tuple[str, str]:
    crop, _, disease = class_name.partition("___")
    return crop, disease


def is_healthy(class_name: str) -> bool:
    return split_class_name(class_name)[1] == "healthy"


def pretty_name(class_name: str) -> str:
    """'Corn_(maize)___Common_rust_' -> 'Corn (maize) – Common rust'."""
    crop, disease = split_class_name(class_name)
    crop = crop.replace("_", " ").replace(" ,", ",").strip()
    disease = disease.replace("_", " ").strip()
    return f"{crop} – {disease[:1].upper()}{disease[1:]}"


def class_names_for(task: str) -> list[str]:
    if task == "multiclass":
        return list(PLANTVILLAGE_CLASSES)
    if task == "binary":
        return list(BINARY_CLASSES)
    raise ValueError(f"Unknown task {task!r}, expected one of {TASKS}")


def target_for(class_name: str, task: str) -> int:
    if task == "multiclass":
        return PLANTVILLAGE_CLASSES.index(class_name)
    if task == "binary":
        return 0 if is_healthy(class_name) else 1
    raise ValueError(f"Unknown task {task!r}, expected one of {TASKS}")
