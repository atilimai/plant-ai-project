"""Check the things RELEASE_CHECKLIST.md claims, instead of trusting the ticked boxes.

    python scripts/check_release.py

Exits non-zero if any check fails. Run it before tagging a release; CI runs the
leakage audit separately.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = ["multiclass_mobilenet_v2", "multiclass_efficientnet_b0", "binary_mobilenet_v2", "binary_efficientnet_b0"]
# Strings that only appear in unfinished text. "placeholder" itself is not on the list:
# several documents legitimately talk about the placeholders that were removed.
PLACEHOLDERS = [
    "to be determined", "to be computed", "to be filled", "tbd", "todo",
    "lorem ipsum", "coming soon", "not yet implemented", "fill me", "???",
]
LINK = re.compile(r"\[[^\]]*\]\(([^)#]+)(?:#[^)]*)?\)")

results: list[tuple[bool, str]] = []


def check(ok: bool, message: str) -> None:
    results.append((bool(ok), message))


def markdown_files() -> list[Path]:
    return [p for p in REPO.rglob("*.md")
            if ".git" not in p.parts and ".venv" not in p.parts and "release" not in p.parts]


def check_links() -> None:
    broken = []
    for doc in markdown_files():
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (doc.parent / target).exists():
                broken.append(f"{doc.relative_to(REPO).as_posix()} -> {target}")
    check(not broken, f"local markdown links resolve ({len(broken)} broken: {broken[:5]})")


def check_placeholders() -> None:
    hits = []
    skip = {"scripts/check_release.py", "docs/previous_state_audit.md", "CONTRIBUTING.md"}
    for doc in markdown_files():
        rel = doc.relative_to(REPO).as_posix()
        if rel in skip:
            continue
        text = doc.read_text(encoding="utf-8").lower()
        text = text.replace("license_placeholder.md", "")  # the file we removed, referred to by name
        hits += [f"{rel}: {word}" for word in PLACEHOLDERS if word in text]
    check(not hits, f"no placeholder text left in docs ({hits[:5]})")


def check_audit() -> None:
    audit_path = REPO / "data" / "splits" / "leakage_audit.json"
    if not audit_path.exists():
        check(False, "data/splits/leakage_audit.json exists")
        return
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    check(audit.get("passed") is True, "leakage audit passed")
    check(not any(v for per_key in audit["overlaps"].values() for v in per_key.values()),
          "no leaf group, leaf id, camera frame or file hash is shared between splits")
    check(audit.get("frame_buffer_violations") == 0, "no frame-buffer violations")


def check_artifacts() -> None:
    for run in RUNS:
        reports = REPO / "artifacts" / "reports" / run
        figures = REPO / "artifacts" / "figures" / run
        check((reports / "test" / "metrics.json").exists(), f"{run}: test metrics exported")
        check((reports / "test" / "per_class_metrics.csv").exists(), f"{run}: per-class metrics exported")
        check((reports / "failure_analysis.md").exists(), f"{run}: failure analysis written")
        check((figures / "confusion_matrix_test_normalized.png").exists(), f"{run}: confusion matrix exported")
        check((figures / "gradcam_errors.jpg").exists(), f"{run}: Grad-CAM figures exported")
        check((REPO / "artifacts" / "sample_outputs" / run / "prediction_gallery.jpg").exists(),
              f"{run}: prediction gallery exported")
    check((REPO / "artifacts" / "reports" / "summary.md").exists(), "results summary written")


def check_metrics_are_real() -> None:
    """Test metrics must come from the audited test split and cover every class."""
    for run in RUNS:
        path = REPO / "artifacts" / "reports" / run / "test" / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        check(m["split"] == "test" and m["variant"] == "color", f"{run}: metrics are from the colour test split")
        expected = 38 if m["task"] == "multiclass" else 2
        check(len(m["per_class"]) == expected, f"{run}: {expected} classes reported")
        check(all(c["support"] > 0 for c in m["per_class"]), f"{run}: every class has test images")


def check_model_card() -> None:
    card = REPO / "MODEL_CARD.md"
    if not card.exists():
        check(False, "MODEL_CARD.md exists")
        return
    text = card.read_text(encoding="utf-8")
    check("cc-by-sa" in text.lower() or "CC BY-SA" in text, "model card states the licence")
    check("leaf" in text.lower() and "leak" in text.lower(), "model card explains the leaf-level split")
    check("field" in text.lower(), "model card warns about field images")
    summary = REPO / "artifacts" / "reports" / "summary.json"
    if summary.exists():
        rows = json.loads(summary.read_text(encoding="utf-8"))
        headline = next((r for r in rows if r["run"] == "multiclass_mobilenet_v2"
                         and r["split"] == "test" and r["variant"] == "color"), None)
        if headline:
            # Accept either "0.9932" or "99.3%" as the way the number is written.
            written = [f"{headline['accuracy']:.4f}", f"{headline['accuracy'] * 100:.1f}"]
            check(any(form in text for form in written),
                  f"model card quotes the measured test accuracy ({written[0]})")


def check_repository_clean() -> None:
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split("\n")
    images = [f for f in tracked
              if f.startswith(("data/raw/", "data/interim/", "data/processed/"))
              and Path(f).name not in {".gitkeep", "README.md"}]
    check(not images, f"no dataset files committed ({images[:3]})")
    weights = [f for f in tracked if f.endswith((".pt", ".pth", ".onnx", ".safetensors"))]
    check(not weights, f"no model weights committed ({weights[:3]})")
    check((REPO / "LICENSE").exists() and (REPO / "LICENSING.md").exists(), "licence files present")
    check(not (REPO / "LICENSE_PLACEHOLDER.md").exists(), "licence placeholder removed")


def main() -> int:
    check_links()
    check_placeholders()
    check_audit()
    check_artifacts()
    check_metrics_are_real()
    check_model_card()
    check_repository_clean()

    failed = [m for ok, m in results if not ok]
    for ok, message in results:
        print(f"[{'ok' if ok else 'FAIL'}] {message}")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
