# src/evaluation/

This directory will contain the evaluation pipeline, metrics computation, and report generation utilities.

## Planned Contents

| Module | Purpose |
|---|---|
| `evaluator.py` | Evaluation runner that loads a checkpoint and runs inference on the test set |
| `metrics.py` | Per-class precision, recall, F1, macro/weighted aggregates |
| `reporter.py` | Exports evaluation results as structured files (CSV/JSON) to `artifacts/reports/` |

## Key Constraints

- Evaluation must always be run on the **held-out test set**, never the validation set.
- The test set must be verified as leakage-free (via `leaf_id` audit) before evaluation.
- Evaluation results must not be used to tune hyperparameters; that role belongs to the validation set.
- No implementation code exists in this directory yet.

## Status

Empty placeholder. Implementation begins in Week 3 (see `ROADMAP.md`).
