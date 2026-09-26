"""Export a training checkpoint for release.

    python -m src.inference.export --checkpoint models/checkpoints/multiclass_mobilenet_v2/best.pt

Writes models/exported/<run>/ with
  model.safetensors   weights only (no pickle, safe to load from the Hub)
  config.json         architecture, class names and preprocessing
  model.onnx          optional (--onnx), checked against PyTorch on random inputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.inference.predictor import load_checkpoint
from src.utils.config import display_path, resolve_path


def export(checkpoint: str | Path, out_dir: str | Path | None = None, onnx: bool = False) -> Path:
    from safetensors.torch import save_file

    checkpoint = resolve_path(checkpoint)
    model, meta = load_checkpoint(checkpoint, "cpu")
    run = checkpoint.parent.name
    out_dir = resolve_path(out_dir or f"models/exported/{run}")
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, out_dir / "model.safetensors")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = {
        **meta,
        "run_name": run,
        "epoch": ckpt.get("epoch"),
        "best_val_macro_f1": ckpt.get("best_metric"),
        "framework": "pytorch",
        "torch_version": torch.__version__.split("+")[0],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if onnx:
        export_onnx(model, meta["image_size"], out_dir / "model.onnx")
    print(f"Exported {run} -> {display_path(out_dir)}")
    return out_dir


def export_onnx(model: torch.nn.Module, image_size: int, path: Path) -> None:
    """Single-file ONNX with a dynamic batch dimension.

    Left on the exporter's default opset: pinning one makes it run the version
    converter, which fails on these graphs. `external_data=False` keeps the weights
    inside the .onnx file instead of a sidecar, so the release is one file.
    """
    sidecar = path.with_suffix(".onnx.data")
    if sidecar.exists():  # left behind by an older export that stored weights separately
        sidecar.unlink()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, path, dynamo=True, external_data=False,
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
    )
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping the ONNX parity check")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batch = torch.randn(4, 3, image_size, image_size)
    with torch.no_grad():
        expected = model(batch).numpy()
    actual = session.run(None, {"pixel_values": batch.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff:.2e}")
    print(f"ONNX parity check passed (max abs diff {max_diff:.1e})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args(argv)
    export(args.checkpoint, args.out_dir, args.onnx)


if __name__ == "__main__":
    main()
