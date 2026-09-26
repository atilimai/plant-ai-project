"""Image pipelines.

Augmentation choices, and why (see also docs/augmentation.md):

* Flips and 90° rotations only. Leaves are photographed flat on a table, so any
  orientation is plausible, and these transforms add no padding. RandomRotation
  with arbitrary angles would paint black corners the model could learn from.
* RandomResizedCrop with a mild scale range (0.6-1.0). Lesions are often small;
  aggressive crops can cut the only symptom out and effectively mislabel the image.
* Colour jitter on brightness/contrast/saturation, with a very small hue range.
  Hue carries disease signal (chlorosis, rust, necrosis), so we barely touch it.

Validation and test images only get a resize + normalisation.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRightAngleRotation(torch.nn.Module):
    def forward(self, img):
        k = int(torch.randint(0, 4, ()).item())
        return T.functional.rotate(img, 90 * k) if k else img


def build_transforms(train: bool, image_size: int = 224, aug: dict | None = None) -> T.Compose:
    aug = aug or {}
    to_tensor = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize((image_size, image_size), antialias=True), *to_tensor])

    ops = [
        T.RandomResizedCrop(image_size, scale=tuple(aug.get("crop_scale", (0.6, 1.0))), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ]
    if aug.get("right_angle_rotation", True):
        ops.append(RandomRightAngleRotation())
    jitter = aug.get("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02})
    if jitter:
        ops.append(T.RandomApply([T.ColorJitter(**jitter)], p=aug.get("color_jitter_p", 0.8)))
    return T.Compose([*ops, *to_tensor])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation for display; works on CHW or NCHW tensors."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(-1, 1, 1)
    return (t * std + mean).clamp(0, 1)
