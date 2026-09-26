# src/models

`factory.py` is the whole module.

* `build_model(arch, num_classes)` — torchvision MobileNetV2 or EfficientNet-B0 with ImageNet
  weights and a fresh `Dropout + Linear` head. Both backbones expose `features`, which keeps the
  rest of the code architecture-agnostic.
* `gradcam_layer(model)` — the last convolutional block, 7×7 at 224 px input: the deepest layer
  that still has spatial resolution, so the most class-specific map we can put back on the image.
* `parameter_groups(...)` — splits parameters into backbone/head and decay/no-decay. Pretrained
  layers get a smaller learning rate; norm layers and biases get no weight decay.
* `set_backbone_trainable(...)` — used by `train.freeze_backbone_epochs` for head-only warm-up.

Adding an architecture means adding one entry to `ARCHITECTURES`, as long as it follows the
`features` + `classifier[-1]` layout; anything else also needs a `gradcam_layer` branch.
