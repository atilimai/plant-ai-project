"""Grad-CAM (Selvaraju et al., 2017) with plain forward/backward hooks.

We hook ``model.features[-1]`` (see ``src.models.factory.gradcam_layer``): the last
conv block of MobileNetV2 / EfficientNet-B0, a 7x7 map at 224px input. Writing the
~40 lines ourselves avoids a dependency and makes it easy to batch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations = None
        self._gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def remove(self):
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, images: torch.Tensor, targets: list[int] | torch.Tensor | None = None):
        """Return (cams, probs). cams is N x H x W in [0, 1] at the input resolution.

        ``targets`` defaults to the predicted class of each image.
        """
        was_training = self.model.training
        self.model.eval()
        images = images.detach().requires_grad_(True)  # hooks need a graph even if weights are frozen
        with torch.enable_grad():
            logits = self.model(images)
            probs = logits.softmax(dim=1).detach()
            if targets is None:
                targets = logits.argmax(dim=1)
            targets = torch.as_tensor(targets, device=logits.device).view(-1, 1)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, targets).sum().backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(dim=1).values, flat.max(dim=1).values
        span = torch.where(hi > lo, hi - lo, torch.ones_like(hi))  # an all-zero map stays zero
        cam = (cam - lo[:, None, None]) / span[:, None, None]
        self.model.train(was_training)
        return cam.detach().cpu().numpy(), probs.cpu().numpy()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap onto an HxWx3 float image in [0, 1]."""
    heat = colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def leaf_focus(cam: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Share of Grad-CAM mass that falls on the leaf (mask from the segmented image).

    Close to the leaf's area share means the map is spread evenly; clearly below it
    means the model is looking at background.
    """
    total = cam.sum()
    return float((cam * leaf_mask).sum() / total) if total > 0 else float("nan")


def plot_gradcam_triplet(rgb: np.ndarray, cam: np.ndarray, title: str | None = None, axes=None):
    """Original / heatmap / overlay, the layout asked for in issue #08."""
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    else:
        fig = axes[0].figure
    axes[0].imshow(rgb)
    axes[0].set_title("Image", fontsize=9)
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM", fontsize=9)
    axes[2].imshow(overlay(rgb, cam))
    axes[2].set_title("Overlay", fontsize=9)
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig
