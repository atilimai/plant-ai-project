"""Train a classifier from a YAML config.

    python -m src.training.train --config configs/multiclass_mobilenet_v2.yaml
    python -m src.training.train --config configs/binary_mobilenet_v2.yaml train.epochs=2
"""

import argparse

from src.training.trainer import train
from src.utils.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = parser.parse_args(argv)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
