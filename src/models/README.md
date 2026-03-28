# src/models/

This directory will contain model architecture definitions and fine-tuning wrappers.

## Planned Contents

| Module | Purpose |
|---|---|
| `mobilenetv2.py` | MobileNetV2 fine-tuning wrapper for binary and multiclass classification |
| `efficientnet_b0.py` | EfficientNet B0 fine-tuning wrapper for binary and multiclass classification |
| `factory.py` | Model factory function to instantiate models by name from config |

## Design Notes

- Both model wrappers will support a configurable number of output classes
  (2 for binary, N for multiclass).
- The final classification head will replace the pre-trained classifier layer.
- Pre-trained weights will be loaded from `torchvision.models` or `timm`.
- No implementation code exists in this directory yet.

## Status

Empty placeholder. Implementation begins in Week 2 (see `ROADMAP.md`).
