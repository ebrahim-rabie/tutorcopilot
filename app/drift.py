"""Phase 3 (P3.5): drift monitoring — trend check against a pinned baseline.

The gate (``app/eval_gate.py``) hard-fails on *any* regression vs the previous
report, one run at a time. Drift answers the complementary question: has the
training data or the eval surface moved materially away from an **accepted,
pinned** baseline (``evals/drift_baseline.json``)?

Contract (pure functions only — no filesystem, no network):

* ``snapshot`` flattens export health metrics (``FinetuneEvalRunner.
  data_metrics``) plus the gate's compared metrics into ``{label: number}``.
* ``drift_findings`` reports only **unfavourable** movements — a decline for
  higher-is-better metrics, a rise for ``LOWER_IS_BETTER`` ones — at or beyond
  a warn/fail relative band (default 10% / 25%). Improvements and new metrics
  that the baseline predates are not drift.
* Baselines are pinned by hand (``--update-baseline``) so the comparison does
  not chase itself the way the run-to-run gate does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.eval_gate import COMPARE, dig

DRIFT_WARN_RATIO = 0.10
DRIFT_FAIL_RATIO = 0.25

# Metrics where *more* is worse: drift flags increases, not decreases.
LOWER_IS_BETTER = {
    "export.dedup_ratio",
    "export.failed_generation_ratio",
}


def snapshot(
    data_metrics: dict[str, Any], gate_report: dict | None = None
) -> dict[str, float]:
    """Flatten export + gate metrics into ``{label: number}`` for comparison."""
    out: dict[str, float] = {}
    for key, value in data_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[f"export.{key}"] = float(value)
    if gate_report:
        for label, path in COMPARE:
            value = dig(gate_report, path)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            out[label] = float(value)
    return out


def drift_findings(
    current: dict[str, float],
    baseline: dict[str, float],
    warn_ratio: float = DRIFT_WARN_RATIO,
    fail_ratio: float = DRIFT_FAIL_RATIO,
) -> list[dict]:
    """Unfavourable movements of ``current`` vs ``baseline``, worst first.

    Each finding: ``{metric, baseline, current, change_ratio, level}`` with
    ``level`` in ``{"warn", "fail"}``. Metrics the baseline predates, non-
    numeric values, and zero baselines (undefined relative change) are skipped.
    """
    findings: list[dict] = []
    for label in sorted(current):
        if label not in baseline:
            continue
        cur = current[label]
        base = baseline[label]
        if isinstance(cur, bool) or isinstance(base, bool):
            continue
        if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
            continue
        if base == 0:
            continue
        delta = cur - base
        unfavourable = delta > 0 if label in LOWER_IS_BETTER else delta < 0
        if not unfavourable:
            continue
        ratio = abs(delta) / abs(base)
        if ratio < warn_ratio:
            continue
        findings.append(
            {
                "metric": label,
                "baseline": base,
                "current": cur,
                "change_ratio": round(ratio, 4),
                "level": "fail" if ratio >= fail_ratio else "warn",
            }
        )
    findings.sort(key=lambda f: f["change_ratio"], reverse=True)
    return findings


def load_drift_baseline(path: str | Path) -> dict | None:
    """Baseline dict, or None if missing or unreadable.

    Callers must treat "file exists but returned None" as a failure: a corrupt
    baseline must not silently disable drift checking.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def write_drift_baseline(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
