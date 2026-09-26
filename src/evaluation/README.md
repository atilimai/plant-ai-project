# src/evaluation

| Module | What it does |
|---|---|
| `evaluate.py` | Runs a checkpoint over a split and writes metrics, predictions and figures |
| `metrics.py` | Per-class precision/recall/F1, macro and weighted averages, balanced accuracy, top-k, ECE, binary sensitivity/specificity/ROC-AUC |
| `failure_analysis.py` | Error breakdown, prediction gallery, Grad-CAM figures, Grad-CAM leaf-focus study |
| `summarize.py` | Collects every run's `metrics.json` into `artifacts/reports/summary.md` |

```bash
python -m src.evaluation.evaluate --checkpoint models/checkpoints/multiclass_mobilenet_v2/best.pt
python -m src.evaluation.evaluate --checkpoint ... --variant segmented   # background removed
python -m src.evaluation.failure_analysis --checkpoint ...
```

Guard rails:

* evaluating on `test` aborts unless `data/splits/leakage_audit.json` says `"passed": true`;
* nothing here touches the training or validation data for anything other than reporting;
* multiclass runs also report healthy-vs-diseased metrics derived from the 38-class probabilities,
  which is what the binary models are compared against.

Outputs land in `artifacts/reports/<run>/<split>/` and `artifacts/figures/<run>/`.
