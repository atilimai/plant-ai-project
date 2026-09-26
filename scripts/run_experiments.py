"""Train, evaluate, analyse and export every experiment config in one go.

    python scripts/run_experiments.py                       # all four runs
    python scripts/run_experiments.py configs/multiclass_mobilenet_v2.yaml
    python scripts/run_experiments.py --skip-train          # re-evaluate existing checkpoints

Each stage skips work that is already done (training resumes from last.pt), so the
script can simply be re-run after a Colab disconnect.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation import failure_analysis  # noqa: E402
from src.evaluation.evaluate import evaluate  # noqa: E402
from src.evaluation.summarize import summarize_runs  # noqa: E402
from src.inference.export import export  # noqa: E402
from src.training.trainer import train  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/multiclass_mobilenet_v2.yaml",
    "configs/multiclass_efficientnet_b0.yaml",
    "configs/binary_mobilenet_v2.yaml",
    "configs/binary_efficientnet_b0.yaml",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--override", action="append", default=[], help="config override, repeatable")
    args = parser.parse_args(argv)

    for config_path in args.configs:
        cfg = load_config(REPO / config_path, [f"data.num_workers={args.num_workers}", *args.override])
        run_dir = resolve_path(cfg["output_dir"]) / cfg["run_name"]
        print(f"\n=== {cfg['run_name']} ===")
        if not args.skip_train:
            train(cfg)

        best = run_dir / "best.pt"
        # Checkpoints are not committed; keep the training log and resolved config with the reports.
        report_dir = REPO / "artifacts" / "reports" / cfg["run_name"]
        report_dir.mkdir(parents=True, exist_ok=True)
        for name in ("history.csv", "config.yaml"):
            shutil.copy2(run_dir / name, report_dir / name)

        common = dict(num_workers=args.num_workers, batch_size=128)
        evaluate(best, split="val", **common)
        evaluate(best, split="test", **common)
        # Same test photos, background removed (black) and background flattened to the
        # colour of the lab table: two different ways of asking what the background contributes.
        evaluate(best, split="test", variant="segmented", **common)
        evaluate(best, split="test", variant="segmented_gray", **common)
        failure_analysis.main(["--checkpoint", str(best)])
        export(best, onnx=True)

    summarize_runs()


if __name__ == "__main__":
    main()
