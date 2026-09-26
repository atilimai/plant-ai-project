# src/training

`trainer.py` holds the loop, `train.py` is the CLI:

```bash
python -m src.training.train --config configs/multiclass_mobilenet_v2.yaml
python -m src.training.train --config configs/binary_mobilenet_v2.yaml train.epochs=3 data.batch_size=32
```

What the loop does: AdamW with separate backbone/head learning rates, linear warm-up into cosine
decay per step, label smoothing, optional class weighting, gradient clipping, AMP and
`channels_last` on CUDA, early stopping on validation macro-F1.

Each run writes `models/checkpoints/<run_name>/`:

| File | Contents |
|---|---|
| `best.pt` / `last.pt` | Weights, optimizer, scheduler, scaler, epoch, full history, config and the metadata inference needs (arch, class names, image size, normalisation) |
| `history.csv` | One row per epoch |
| `config.yaml` | The resolved config, after inheritance and CLI overrides |

Checkpoints are written to a temporary file and renamed, so a Colab disconnect mid-save cannot
leave a corrupt file. Re-running the same command resumes from `last.pt`; a run that started on CPU
can continue on GPU.

Validation metrics are for model selection only. Test numbers come from `src/evaluation`.
