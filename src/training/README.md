# src/training/

This directory will contain the training loop, optimizer setup, and training configuration utilities.

## Planned Contents

| Module | Purpose |
|---|---|
| `trainer.py` | Main training loop with logging, checkpointing, and early stopping |
| `optimizer.py` | Optimizer and learning rate scheduler factory |
| `callbacks.py` | Training callbacks (optional, e.g., checkpoint saver, metric logger) |

## Design Notes

- The training loop will support both binary and multiclass classification tracks
  with minimal configuration changes.
- Checkpoints will be saved to `models/checkpoints/`.
- Experiment hyperparameters will be loaded from `configs/` YAML files.
- Training is designed to run on Google Colab (T4 or A100 GPU).
- No implementation code exists in this directory yet.

## Status

Empty placeholder. Implementation begins in Week 2 (see `ROADMAP.md`).
