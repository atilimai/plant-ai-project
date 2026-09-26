---
title: "Design and implement evaluation metrics pipeline"
labels: ["task"]
---

## Summary

Implement an evaluation pipeline in `src/evaluation/` that computes per-class precision, recall, F1, and macro/weighted aggregates for both the binary and multiclass classification tracks. The pipeline should output results to `artifacts/reports/` in a reproducible format.

## Why It Matters

Aggregate accuracy alone is insufficient for plant disease classification due to class imbalance and the practical importance of minimizing false negatives on diseased classes. Per-class metrics provide the granular view needed for responsible model evaluation.

## Acceptance Criteria

- [x] Evaluation runner implemented in `src/evaluation/`
- [x] Per-class precision, recall, and F1 computed for both classification tracks
- [x] Macro-averaged and weighted F1 computed
- [x] Support confidence score or probability output (for potential threshold tuning)
- [x] Results exported as structured files (CSV or JSON) to `artifacts/reports/`
- [x] Evaluation is run only on the held-out test set (not validation)
- [x] Evaluation results referenced and summarized in the model card

## Dependencies

- Issue #02: Leakage guard in place — evaluation must use a properly partitioned test set
- Issue #03 or #04: At least one trained model checkpoint available

## Notes

- Per-class metrics are the primary evaluation artifact for the model card
- Consider adding top-k accuracy for the multiclass track
- Evaluation runner should be reusable across both classification tracks with minimal changes
- All evaluation code should be callable from a Colab notebook

---

## Status

Done. `src/evaluation/metrics.py` and `evaluate.py`: per-class precision/recall/F1, macro and weighted averages, balanced accuracy, top-3/top-5, ECE, and for the binary track sensitivity, specificity and ROC-AUC. Probabilities and per-image predictions are exported, so thresholds can be studied afterwards. Test-set evaluation refuses to run unless the leakage audit passed.
