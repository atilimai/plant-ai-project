"""Transfer-learning backbones.

Both architectures come from torchvision so there is a single, well-tested source
of ImageNet weights and no extra dependency. The ImageNet classifier is replaced
by dropout + a fresh linear layer sized for the task.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

ARCHITECTURES = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")
    constructor, weights = ARCHITECTURES[arch]
    model = constructor(weights=weights if pretrained else None)

    # Both torchvision models end in Sequential(Dropout, Linear).
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return model


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block: 7x7 feature map at 224px input for both backbones.

    It is the deepest layer that still has spatial resolution, so its activations are
    the most class-specific ones we can still map back onto the image.
    """
    return model.features[-1]


def parameter_groups(model: nn.Module, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
    """Split params into backbone/head and decay/no-decay (norm layers and biases)."""
    groups = {
        (part, decay): [] for part in ("backbone", "head") for decay in (True, False)
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        part = "head" if name.startswith("classifier") else "backbone"
        decay = param.ndim > 1
        groups[(part, decay)].append(param)

    out = []
    for (part, decay), params in groups.items():
        if params:
            out.append({
                "params": params,
                "lr": lr * (backbone_lr_mult if part == "backbone" else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
                "name": f"{part}{'' if decay else '_no_decay'}",
            })
    return out


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def smoke_test(arch: str, num_classes: int = 3) -> torch.Size:
    model = build_model(arch, num_classes, pretrained=False).eval()
    return model(torch.zeros(1, 3, 224, 224)).shape
