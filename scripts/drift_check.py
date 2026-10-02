"""Phase 3 (P3.5): drift check CLI.

Compares the current export + gate metrics against a pinned baseline
(``evals/drift_baseline.json``) and prints WARN/FAIL lines for unfavourable
movements beyond a relative band (default 10% / 25%).

Exit codes:
  0  no drift, only WARN-level drift (unless --strict), or baseline established
  1  FAIL-level drift, unreadable baseline/report, or (with --strict) any warn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.drift import (
    DRIFT_FAIL_RATIO,
    DRIFT_WARN_RATIO,
    drift_findings,
    load_drift_baseline,
    snapshot,
    write_drift_baseline,
)
from app.eval_finetune import FinetuneEvalRunner
from app.eval_gate import load_gate_report
from app.settings import Settings

JUNK_GENERATION_FLOOR = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="evals/drift_baseline.json",
        help="Pinned drift baseline (default: %(default)s)",
    )
    parser.add_argument(
        "--gate-report",
        default="evals/reports/gate_latest.json",
        help="Latest gate report to fold eval metrics in (default: %(default)s)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Accept the current metrics as the new pinned baseline",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit 1 on WARN-level drift too"
    )
    parser.add_argument("--warn", type=float, default=DRIFT_WARN_RATIO)
    parser.add_argument("--fail", type=float, default=DRIFT_FAIL_RATIO)
    args = parser.parse_args(argv)

    settings = Settings()
    metrics = FinetuneEvalRunner(settings).data_metrics()

    gate_path = Path(args.gate_report)
    gate_report = None
    if gate_path.exists():
        gate_report = load_gate_report(gate_path)
        if gate_report is None:
            print(f"DRIFT: gate report unreadable: {gate_path}")
            return 1
    else:
        print(f"note: no gate report at {gate_path}; checking export metrics only")

    current = snapshot(metrics, gate_report)
    baseline_path = Path(args.baseline)

    if args.update_baseline:
        write_drift_baseline(baseline_path, current)
        print(f"drift baseline updated: {baseline_path} ({len(current)} metrics)")
        return 0

    baseline = load_drift_baseline(baseline_path)
    if baseline is None:
        if baseline_path.exists():
            print(f"DRIFT: baseline unreadable: {baseline_path}")
            return 1
        write_drift_baseline(baseline_path, current)
        print(
            f"baseline established (first drift run): {baseline_path} "
            f"({len(current)} metrics)"
        )
        return 0

    findings = drift_findings(current, baseline, args.warn, args.fail)
    for f in findings:
        pct = (f["current"] - f["baseline"]) / abs(f["baseline"]) * 100
        print(
            f"{f['level'].upper()} drift {f['metric']}: "
            f"{f['baseline']} -> {f['current']} ({pct:+.1f}%)"
        )

    junk_ratio = current.get("export.failed_generation_ratio")
    junk_warned = junk_ratio is not None and junk_ratio > JUNK_GENERATION_FLOOR
    if junk_warned:
        print(
            f"WARN data quality: {junk_ratio:.0%} of exported generations start "
            "with 'LLM call failed' — regenerate traces with TC_LLM_API_KEY "
            "before training"
        )

    n_fail = sum(1 for f in findings if f["level"] == "fail")
    n_warn = len(findings) - n_fail
    if not findings:
        print(f"no drift: {len(current)} metrics within tolerance of baseline")

    failed = n_fail > 0 or (args.strict and (n_warn > 0 or junk_warned))
    print(f"DRIFT: {'FAIL' if failed else 'PASS'} (warn={n_warn} fail={n_fail})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
