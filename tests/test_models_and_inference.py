import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import PlantVillageDataset, load_split, make_loader
from src.data.labels import BINARY_CLASSES, class_names_for, is_healthy, pretty_name, target_for
from src.data.transforms import build_transforms, denormalize
from src.inference.predictor import Predictor, load_checkpoint
from src.models.factory import build_model, gradcam_layer, parameter_groups
from src.training.trainer import class_weights, warmup_cosine
from src.utils.config import load_config
from src.visualization.grad_cam import GradCAM, leaf_focus, overlay


@pytest.mark.parametrize("arch", ["mobilenet_v2", "efficientnet_b0"])
def test_model_output_shape_and_gradcam(arch):
    model = build_model(arch, num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    assert model.eval()(x).shape == (2, 5)
    with GradCAM(model, gradcam_layer(model)) as cam:
        maps, probs = cam(x)
    assert maps.shape == (2, 224, 224)
    assert maps.min() >= 0 and maps.max() <= 1
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)


def test_parameter_groups_cover_every_trainable_parameter():
    model = build_model("mobilenet_v2", 3, pretrained=False)
    groups = parameter_groups(model, lr=1e-3, backbone_lr_mult=0.1, weight_decay=0.05)
    n_grouped = sum(p.numel() for g in groups for p in g["params"])
    assert n_grouped == sum(p.numel() for p in model.parameters())
    head = next(g for g in groups if g["name"] == "head")
    backbone = next(g for g in groups if g["name"] == "backbone")
    assert head["lr"] == pytest.approx(1e-3) and backbone["lr"] == pytest.approx(1e-4)
    assert all(g["weight_decay"] == 0 for g in groups if g["name"].endswith("no_decay"))


def test_labels():
    assert len(class_names_for("multiclass")) == 38
    assert sum(is_healthy(c) for c in class_names_for("multiclass")) == 12
    assert class_names_for("binary") == BINARY_CLASSES
    assert target_for("Tomato___healthy", "binary") == 0
    assert target_for("Tomato___Late_blight", "binary") == 1
    assert pretty_name("Corn_(maize)___Common_rust_") == "Corn (maize) – Common rust"


def test_class_weights_and_schedule():
    w = class_weights(torch.tensor([100, 400]), "inverse_sqrt")
    assert w[0] > w[1] and w.sum().item() == pytest.approx(2.0)
    assert class_weights(torch.tensor([1, 2]), "none") is None
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(99) < 0.05


def test_eval_transform_is_deterministic_and_train_transform_is_not():
    img = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))
    eval_tf = build_transforms(train=False, image_size=32)
    assert torch.equal(eval_tf(img), eval_tf(img))
    assert eval_tf(img).shape == (3, 32, 32)
    train_tf = build_transforms(train=True, image_size=32)
    torch.manual_seed(0)
    assert not all(torch.equal(train_tf(img), train_tf(img)) for _ in range(5))
    assert denormalize(eval_tf(img)).min() >= 0


def test_dataset_reads_both_variants(image_tree):
    root, manifest_path = image_tree
    manifest = load_split(manifest_path, "train")
    ds = PlantVillageDataset(manifest, root, task="binary", transform=build_transforms(False, 32))
    image, target = ds[0]
    assert image.shape == (3, 32, 32) and target in (0, 1)
    seg = PlantVillageDataset(manifest, root, task="multiclass", variant="segmented")
    assert len(seg) == len(ds)
    loader = make_loader(manifest, root, "multiclass", train=True, image_size=32, batch_size=4, num_workers=0)
    images, targets = next(iter(loader))
    assert images.shape == (4, 3, 32, 32)


def test_predictor_round_trip_from_checkpoint_and_export(tmp_path, image_tree):
    from safetensors.torch import save_file

    root, manifest_path = image_tree
    model = build_model("mobilenet_v2", 2, pretrained=False).eval()
    meta = {"arch": "mobilenet_v2", "task": "binary", "class_names": BINARY_CLASSES,
            "image_size": 64, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "dropout": 0.2}
    torch.save({"model_state": model.state_dict(), "meta": meta}, tmp_path / "best.pt")

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    save_file(model.state_dict(), export_dir / "model.safetensors")
    (export_dir / "config.json").write_text(json.dumps(meta))

    image = next((root / "color").rglob("*.JPG"))
    results = []
    for source in (tmp_path / "best.pt", export_dir):
        reloaded, reloaded_meta = load_checkpoint(source)
        assert reloaded_meta["class_names"] == BINARY_CLASSES
        predictor = Predictor.load(source, device="cpu")
        results.append(predictor.predict_proba([image, image]))
        pred, cam, blended = predictor.explain(image)
        assert pred.label in BINARY_CLASSES and cam.shape == (64, 64) and blended.shape == (64, 64, 3)
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)


def test_overlay_and_leaf_focus():
    rgb = np.zeros((4, 4, 3))
    cam = np.zeros((4, 4))
    cam[:2] = 1.0
    mask = np.zeros((4, 4))
    mask[:2] = 1
    assert overlay(rgb, cam).shape == (4, 4, 3)
    assert leaf_focus(cam, mask) == pytest.approx(1.0)
    assert leaf_focus(cam, 1 - mask) == pytest.approx(0.0)


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("train:\n  epochs: 10\n  lr: 0.001\ndata:\n  batch_size: 64\n")
    (tmp_path / "exp.yaml").write_text("base: base.yaml\nrun_name: x\ntrain:\n  epochs: 3\n")
    cfg = load_config(tmp_path / "exp.yaml", ["data.batch_size=8", "model.arch=efficientnet_b0"])
    assert cfg["train"] == {"epochs": 3, "lr": 0.001}
    assert cfg["data"]["batch_size"] == 8
    assert cfg["model"]["arch"] == "efficientnet_b0"
    with pytest.raises(ValueError):
        load_config(tmp_path / "exp.yaml", ["no-equals-sign"])
