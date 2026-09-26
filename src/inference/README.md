# src/inference

| Module | What it does |
|---|---|
| `predictor.py` | `Predictor.load(path)` accepts a training checkpoint or an exported folder; `predict`, `predict_proba`, `explain` (prediction + Grad-CAM overlay) |
| `export.py` | Writes `model.safetensors` + `config.json` (+ optional ONNX, checked against PyTorch) to `models/exported/<run>/` |
| `predict.py` | CLI for classifying image files |

```bash
python -m src.inference.export --checkpoint models/checkpoints/multiclass_mobilenet_v2/best.pt --onnx
python -m src.inference.predict --model models/exported/multiclass_mobilenet_v2 leaf.jpg
```

The exported folder is self-describing: `config.json` carries the architecture, class names, input
size and normalisation, so nothing about preprocessing has to be guessed at inference time. It is
also exactly what we upload to the Hugging Face Hub, and what `app/app.py` loads.
