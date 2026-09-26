from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from tqdm.auto import tqdm

from src.data.dataset import load_split, make_loader
from src.data.labels import class_names_for
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.metrics import classification_metrics
from src.models.factory import build_model, parameter_groups, set_backbone_trainable
from src.utils.config import display_path, resolve_path
from src.utils.device import pick_device
from src.utils.seed import seed_everything

CHECKPOINT_FORMAT = 1


def class_weights(counts: torch.Tensor, mode: str | None) -> torch.Tensor | None:
    """'inverse_sqrt' softens the imbalance without letting 150-image classes dominate."""
    if not mode or mode == "none":
        return None
    counts = counts.float().clamp(min=1)
    if mode == "inverse_sqrt":
        w = counts.rsqrt()
    elif mode == "inverse":
        w = 1.0 / counts
    else:
        raise ValueError(f"Unknown class weighting {mode!r}")
    return w * len(w) / w.sum()


def warmup_cosine(total_steps: int, warmup_steps: int, min_factor: float = 0.01):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, amp: bool = False, desc: str = "eval"):
    """Run a loader through the model; returns (targets, probabilities, mean loss)."""
    model.eval()
    targets, probs, loss_sum = [], [], 0.0
    for images, y in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
        logits = logits.float()
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        probs.append(logits.softmax(dim=1).cpu())
        targets.append(y.cpu())
    targets = torch.cat(targets).numpy()
    return targets, torch.cat(probs).numpy(), loss_sum / max(1, len(targets))


def checkpoint_meta(cfg: dict, class_names: list[str]) -> dict:
    return {
        "arch": cfg["model"]["arch"],
        "task": cfg["task"],
        "class_names": class_names,
        "image_size": cfg["data"]["image_size"],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "dropout": cfg["model"].get("dropout", 0.2),
    }


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind if Colab dies mid-save


def train(cfg: dict) -> Path:
    seed_everything(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "auto"))
    task = cfg["task"]
    class_names = class_names_for(task)
    data_cfg, train_cfg = cfg["data"], cfg["train"]

    run_dir = resolve_path(cfg.get("output_dir", "models/checkpoints")) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    manifest = load_split(resolve_path(data_cfg["manifest"]))
    fraction = data_cfg.get("subset_fraction", 1.0)
    if fraction < 1.0:
        manifest = (
            manifest.groupby(["split", "class_name"], group_keys=False)
            .sample(frac=fraction, random_state=cfg.get("seed", 42))
            .reset_index(drop=True)
        )
        print(f"Smoke-test mode: using {fraction:.1%} of every split ({len(manifest)} images)")
    loader_args = dict(
        image_root=resolve_path(data_cfg["image_root"]),
        task=task,
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        seed=cfg.get("seed", 42),
    )
    train_loader = make_loader(
        manifest[manifest.split == "train"], train=True, aug=cfg.get("augmentation"), **loader_args
    )
    val_loader = make_loader(manifest[manifest.split == "val"], train=False, **loader_args)

    model = build_model(
        cfg["model"]["arch"],
        len(class_names),
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.2),
    ).to(device)
    use_amp = device.type == "cuda" and train_cfg.get("amp", True)
    # channels_last is usually a win for convnets, but on this torch/cuDNN build it makes the
    # depthwise convolutions in both backbones ~7x slower (723 -> 104 img/s on an RTX 4060), so it
    # is off by default. Measure before turning it on for another GPU or torch version.
    channels_last = device.type == "cuda" and train_cfg.get("channels_last", False)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input size, so autotuning pays off

    weights = class_weights(train_loader.dataset.class_counts(), train_cfg.get("class_weighting"))
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            lr=train_cfg["lr"],
            backbone_lr_mult=train_cfg.get("backbone_lr_mult", 1.0),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
    )
    epochs = train_cfg["epochs"]
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(epochs * steps_per_epoch, int(train_cfg.get("warmup_epochs", 0) * steps_per_epoch)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    monitor = f"val_{train_cfg.get('monitor', 'macro_f1')}"
    patience = train_cfg.get("early_stopping_patience", epochs)
    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0)
    start_epoch, best, history, stale = 0, -math.inf, [], 0

    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if train_cfg.get("resume", True) and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if state["scaler_state"] and use_amp:  # empty when the run started on CPU
            scaler.load_state_dict(state["scaler_state"])
        start_epoch, best, history = state["epoch"], state["best_metric"], state["history"]
        stale = len(history) - 1 - int(np.argmax([h[monitor] for h in history]))
        print(f"Resuming {cfg['run_name']} after epoch {start_epoch} (best {monitor}={best:.4f})")

    print(
        f"{cfg['run_name']}: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val images, "
        f"{len(class_names)} classes, device={device}, amp={use_amp}"
    )

    for epoch in range(start_epoch, epochs):
        set_backbone_trainable(model, epoch >= freeze_epochs)
        model.train()
        t0, seen, correct, loss_sum = time.time(), 0, 0, 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for images, y in bar:
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if train_cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            bar.set_postfix(loss=f"{loss_sum / seen:.4f}", acc=f"{correct / seen:.3f}")

        y_val, p_val, val_loss = predict(model, val_loader, device, amp=use_amp, desc="val")
        val = classification_metrics(y_val, p_val.argmax(1), class_names, probs=p_val)
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[-1]["lr"],
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "val_loss": val_loss,
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_ece": val["ece"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        improved = row[monitor] > best
        if improved:
            best, stale = row[monitor], 0
        else:
            stale += 1

        print(
            f"epoch {epoch + 1:2d} | train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} | "
            f"val loss {val_loss:.4f} acc {val['accuracy']:.4f} macro-F1 {val['macro_f1']:.4f} | "
            f"{row['seconds']:.0f}s{'  *' if improved else ''}"
        )

        state = {
            "format": CHECKPOINT_FORMAT,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_metric": best,
            "history": history,
            "config": cfg,
            "meta": checkpoint_meta(cfg, class_names),
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        write_history(run_dir / "history.csv", history)

        if stale >= patience:
            print(f"No improvement in {monitor} for {patience} epochs, stopping early.")
            break

    print(f"Best {monitor}: {best:.4f} -> {display_path(best_path)}")
    return run_dir


def write_history(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
