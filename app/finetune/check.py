"""Phase 3 (P3.3): hermetic ``--check-only`` validation of LoRA exports.

Asserts ``config/finetune/{train,val}.jsonl`` are trainable **before** any
(heavy) ML dependency is imported: every line must parse, satisfy
``TrainingExample.from_dict``, and pass the purity gate; train and val must
each keep at least one example. Generations starting with "LLM call failed"
are counted as a curation warning (the exporter keeps failed turns on purpose)
— they are schema-valid, so they never fail the check on their own.

No torch/trl/peft/datasets, no network: this is the smokeable-today half of
P3.3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.finetune.schema import TrainingExample

SUSPICIOUS_PREFIX = "LLM call failed"


def check_export_file(path: str | Path) -> dict[str, Any]:
    """Validate one exported JSONL file; errors carry ``file:line``."""
    p = Path(path)
    errors: list[str] = []
    n_lines = n_valid = n_failed = n_suspicious = 0

    if not p.exists():
        return {
            "path": str(p),
            "n_lines": 0,
            "n_valid": 0,
            "n_failed": 0,
            "n_suspicious": 0,
            "errors": [f"{p.name}: file not found"],
        }

    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        n_lines += 1
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{p.name}:{line_no}: invalid JSON ({exc.msg})")
            continue
        try:
            example = TrainingExample.from_dict(data)
        except (ValueError, TypeError) as exc:
            errors.append(f"{p.name}:{line_no}: {exc}")
            continue
        if not example.is_valid():
            errors.append(f"{p.name}:{line_no}: empty question or generation")
            continue
        n_valid += 1
        if not example.success:
            n_failed += 1
        if example.generation.startswith(SUSPICIOUS_PREFIX):
            n_suspicious += 1

    return {
        "path": str(p),
        "n_lines": n_lines,
        "n_valid": n_valid,
        "n_failed": n_failed,
        "n_suspicious": n_suspicious,
        "errors": errors,
    }


def check_exports(train_path: str | Path, val_path: str | Path) -> dict[str, Any]:
    """Validate both export files; ``ok`` means schema-clean and trainable."""
    train = check_export_file(train_path)
    val = check_export_file(val_path)

    problems: list[str] = []
    if train["n_valid"] < 1:
        problems.append(f"{Path(train_path).name}: needs >= 1 training example")
    if val["n_valid"] < 1:
        problems.append(f"{Path(val_path).name}: needs >= 1 validation example")

    return {
        "ok": not train["errors"] and not val["errors"] and not problems,
        "problems": problems,
        "train": train,
        "val": val,
    }
