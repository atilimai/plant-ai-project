from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, following an optional `base:` key, then apply CLI overrides.

    Overrides use dotted keys, e.g. ``train.epochs=3`` or ``data.batch_size=64``.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("base", None)
    if base:
        base_cfg = load_config(path.parent / base)
        cfg = merge(base_cfg, cfg)

    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        set_by_dotted_key(cfg, key, yaml.safe_load(raw))
    return cfg


def merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_by_dotted_key(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def resolve_path(p: str | Path) -> Path:
    """Paths in configs are relative to the repository root unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def display_path(p: str | Path) -> str:
    """Repo-relative path for log messages, so notebook outputs carry no local paths."""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)
