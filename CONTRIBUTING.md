# Contributing

Thanks for your interest. This is a small research repository; the bar is that anyone can reproduce
what it claims.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.data.prepare --config configs/data.yaml   # downloads 2.2 GB, writes data/splits/
pytest
```

The tests run on synthetic images in seconds and need neither the dataset nor a GPU.

## Before opening a pull request

```bash
ruff check src tests scripts app
pytest
python scripts/audit_manifest.py     # only if you touched anything under data/splits or src/data
```

CI runs exactly these three.

## Research integrity rules

These are the ones we will actually push back on:

1. **Splits stay at the leaf level.** Any change to `src/data/grouping.py` or `src/data/splits.py`
   has to keep `python scripts/audit_manifest.py` passing, and changes to the split policy need the
   reasoning in `data/splits/README.md` updated. If you add a grouping rule, add a test for it.
2. **Test-set numbers come from the pipeline.** Report metrics produced by
   `src.evaluation.evaluate` on the `test` split, not numbers typed into a notebook or a document.
   Never tune hyperparameters on the test split; that is what `val` is for.
3. **No made-up artifacts.** Every figure in `artifacts/` must come from a real model on real data.
   (The repository used to contain confusion matrices generated from random noise; see
   `docs/previous_state_audit.md`.)
4. **No dataset files in git.** Manifests and metadata only. `.gitignore` enforces the common cases.
5. **Claims match evidence.** If a result only holds on the lab-condition test set, say so where it
   is stated.

## Style

* PEP 8, 100-column lines, `ruff` settings in `pyproject.toml`.
* Type hints where they help a reader; docstrings on anything non-obvious, explaining *why* rather
  than restating the code.
* New behaviour that can break silently (grouping, splitting, metrics, checkpoint format) needs a
  test.
* Keep notebooks thin: logic goes in `src/`, notebooks call it. Long-running cells stay behind a
  `RUN_*` flag.

## Licence

Code is GPL-2.0-only; PlantVillage data and anything trained on it is CC BY-SA 3.0. See
`LICENSING.md` before adding dependencies or publishing artifacts.
