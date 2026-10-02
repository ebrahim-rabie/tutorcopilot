"""Tests for P3.5 drift monitoring (pure logic + CLI exit codes).

All hermetic: temp episodic dumps, temp baselines, no network, no model
downloads, no gate report required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.drift import (
    drift_findings,
    load_drift_baseline,
    snapshot,
    write_drift_baseline,
)
from app.settings import Settings
from scripts import drift_check

TURN_LINES = [
    json.dumps(
        {
            "user_message": "what is rrf?",
            "assistant_message": "LLM call failed: timeout",
            "tools_used": ["search_knowledge"],
        }
    ),
    json.dumps(
        {
            "user_message": "explain agents",
            "assistant_message": "LLM call failed: connection reset",
            "tools_used": [{"name": "run_python", "success": True}],
        }
    ),
]


def _gate_report() -> dict:
    return {
        "retrieval": {"mrr": 0.9, "hit_rate_at_1": 0.8, "hit_rate_at_3": 1.0},
        "conversation": {"pass_rate": None},
        "finetune": {"replay": {"tool_faithfulness": 1.0}},
    }


def test_snapshot_flattens_export_and_gate():
    metrics = {
        "n_examples": 2,
        "kept_ratio": 0.5,
        "tool_calls_success_rate": None,
        "dedup_ratio": 0.0,
    }
    snap = snapshot(metrics, _gate_report())
    assert snap["export.n_examples"] == 2.0
    assert snap["export.kept_ratio"] == 0.5
    assert "export.tool_calls_success_rate" not in snap  # None is not comparable
    assert snap["retrieval.mrr"] == 0.9
    assert "conversation.pass_rate" not in snap  # skipped metric
    assert snap["finetune.tool_faithfulness"] == 1.0


def test_snapshot_without_gate_report_is_export_only():
    snap = snapshot({"n_examples": 1}, None)
    assert snap == {"export.n_examples": 1.0}


def test_no_drift_when_equal():
    assert drift_findings({"a": 1.0}, {"a": 1.0}) == []


def test_decline_warn_and_fail_bands():
    warn = drift_findings({"retrieval.mrr": 0.89}, {"retrieval.mrr": 1.0})
    assert warn[0]["level"] == "warn"  # -11%
    fail = drift_findings({"retrieval.mrr": 0.7}, {"retrieval.mrr": 1.0})
    assert fail[0]["level"] == "fail"  # -30%
    assert fail[0]["change_ratio"] == 0.3


def test_improvements_are_not_drift():
    assert drift_findings({"retrieval.mrr": 1.0}, {"retrieval.mrr": 0.9}) == []
    # lower-is-better metric moving down is an improvement
    assert drift_findings({"export.dedup_ratio": 0.05}, {"export.dedup_ratio": 0.2}) == []


def test_lower_is_better_increases_flagged():
    d = drift_findings({"export.dedup_ratio": 0.4}, {"export.dedup_ratio": 0.2})
    assert d[0]["level"] == "fail"  # +100%
    j = drift_findings(
        {"export.failed_generation_ratio": 0.5},
        {"export.failed_generation_ratio": 0.2},
    )
    assert j[0]["level"] == "fail"


def test_skips_new_metrics_zero_baseline_and_non_numbers():
    # metric new since baseline, zero baseline, non-numeric values
    assert drift_findings({"new.metric": 1.0}, {"other": 1.0}) == []
    assert drift_findings({"x": 0.5}, {"x": 0.0}) == []
    assert drift_findings({"x": "bad"}, {"x": 1.0}) == []
    assert drift_findings({True: 1.0}, {True: 1.0}) == []  # type: ignore[dict-item]


def test_baseline_roundtrip_and_unreadable():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "drift_baseline.json"
        assert load_drift_baseline(path) is None  # missing
        write_drift_baseline(path, {"a": 1.0})
        assert load_drift_baseline(path) == {"a": 1.0}
        path.write_text("{corrupt", encoding="utf-8")
        assert load_drift_baseline(path) is None  # exists but unreadable


def _cli(tmp: Path, monkeypatch, argv: list[str]) -> int:
    episodic = tmp / "episodic"
    episodic.mkdir(parents=True, exist_ok=True)
    (episodic / "c.jsonl").write_text("\n".join(TURN_LINES) + "\n", encoding="utf-8")
    settings = Settings(episodic_dir=episodic, trace_dir=tmp / "traces")
    monkeypatch.setattr(drift_check, "Settings", lambda: settings)
    return drift_check.main(argv)


def test_cli_establishes_baseline_then_passes(tmp_path, monkeypatch):
    baseline = tmp_path / "drift_baseline.json"
    missing_gate = tmp_path / "gate_report.json"
    argv = ["--baseline", str(baseline), "--gate-report", str(missing_gate)]

    assert _cli(tmp_path, monkeypatch, argv) == 0
    assert baseline.exists()

    assert _cli(tmp_path, monkeypatch, argv) == 0  # no drift vs itself


def test_cli_fails_on_drift(tmp_path, monkeypatch):
    baseline = tmp_path / "drift_baseline.json"
    missing_gate = tmp_path / "gate_report.json"
    argv = ["--baseline", str(baseline), "--gate-report", str(missing_gate)]
    assert _cli(tmp_path, monkeypatch, argv) == 0  # establish

    # inflate the baseline so the current export looks like a collapse
    data = json.loads(baseline.read_text(encoding="utf-8"))
    data["export.n_examples"] = 999.0
    baseline.write_text(json.dumps(data), encoding="utf-8")

    assert _cli(tmp_path, monkeypatch, argv) == 1


def test_cli_corrupt_baseline_fails(tmp_path, monkeypatch):
    baseline = tmp_path / "drift_baseline.json"
    baseline.write_text("{nope", encoding="utf-8")
    argv = ["--baseline", str(baseline), "--gate-report", str(tmp_path / "none.json")]
    assert _cli(tmp_path, monkeypatch, argv) == 1


def test_cli_update_baseline(tmp_path, monkeypatch):
    baseline = tmp_path / "drift_baseline.json"
    argv = [
        "--baseline",
        str(baseline),
        "--gate-report",
        str(tmp_path / "none.json"),
        "--update-baseline",
    ]
    assert _cli(tmp_path, monkeypatch, argv) == 0
    assert baseline.exists()


def test_cli_strict_promotes_warn_and_junk(tmp_path, monkeypatch):
    baseline = tmp_path / "drift_baseline.json"
    missing_gate = tmp_path / "none.json"
    argv = ["--baseline", str(baseline), "--gate-report", str(missing_gate)]
    assert _cli(tmp_path, monkeypatch, argv) == 0  # establish

    # warn-band decline: n_examples 2 vs baseline 2.25 (-11%) passes by default
    data = json.loads(baseline.read_text(encoding="utf-8"))
    data["export.n_examples"] = 2.25
    baseline.write_text(json.dumps(data), encoding="utf-8")
    assert _cli(tmp_path, monkeypatch, argv) == 0
    assert _cli(tmp_path, monkeypatch, [*argv, "--strict"]) == 1

    # standing junk floor (100% failed generations) is a warning, promoted strict
    data["export.n_examples"] = 2.0
    baseline.write_text(json.dumps(data), encoding="utf-8")
    assert _cli(tmp_path, monkeypatch, argv) == 0
    assert _cli(tmp_path, monkeypatch, [*argv, "--strict"]) == 1
