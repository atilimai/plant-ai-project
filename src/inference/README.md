# src/inference/

This directory will contain inference utilities for running predictions with trained model checkpoints.

## Planned Contents

| Module | Purpose |
|---|---|
| `predictor.py` | Loads a checkpoint and runs inference on a single image or batch |
| `export.py` | Exports trained models to ONNX or TorchScript format |

## Design Notes

- The predictor should support loading from local checkpoints (`models/checkpoints/`)
  and from Hugging Face Hub.
- Inference utilities should return both the predicted class label and confidence score.
- Export utilities will save models to `models/exported/`.
- No implementation code exists in this directory yet.

## Status

Empty placeholder. Implementation planned for Week 4 (see `ROADMAP.md`).
