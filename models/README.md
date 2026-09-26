# models

Weights are **not** committed. This folder is where they land locally.

```
models/
├── checkpoints/<run_name>/     training output: best.pt, last.pt, history.csv, config.yaml
└── exported/<run_name>/        release output: model.safetensors, config.json, model.onnx
```

Run names follow `<task>_<arch>`: `multiclass_mobilenet_v2`, `binary_efficientnet_b0`, and so on.

| | `checkpoints/` | `exported/` |
|---|---|---|
| Format | `torch.save` dict (pickle) | safetensors + JSON (+ ONNX) |
| Contains | weights, optimizer, scheduler, scaler, epoch, history, config, metadata | weights and the metadata inference needs |
| Purpose | resume training, evaluate, analyse | Hugging Face release, demo app, other runtimes |

Produced by:

```bash
python -m src.training.train --config configs/multiclass_mobilenet_v2.yaml
python -m src.inference.export --checkpoint models/checkpoints/multiclass_mobilenet_v2/best.pt --onnx
```

`config.json` in an exported folder carries the architecture, class names, input size and
normalisation, so a consumer never has to guess the preprocessing. The released weights are
CC BY-SA 3.0, like the dataset — see `LICENSING.md`.

The published copies live at
<https://huggingface.co/atilimai/plantvillage-leaf-disease-classifier>.
