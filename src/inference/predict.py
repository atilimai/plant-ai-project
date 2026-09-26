"""Classify leaf images from the command line.

    python -m src.inference.predict --model models/exported/multiclass_mobilenet_v2 leaf1.jpg leaf2.jpg
"""

import argparse

from src.inference.predictor import Predictor, display


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="export folder or training checkpoint")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args(argv)

    predictor = Predictor.load(args.model, device=args.device)
    for path, pred in zip(args.images, predictor.predict(args.images, top_k=args.top_k)):
        alternatives = ", ".join(f"{display(label)} {p:.1%}" for label, p in pred.top_k[1:])
        print(f"{path}: {pred.display_name} ({pred.confidence:.1%})" + (f" | then: {alternatives}" if alternatives else ""))


if __name__ == "__main__":
    main()
