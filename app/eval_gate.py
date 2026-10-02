"""Phase 3 (P3.2): the eval gate — diff two gate reports, fail on regression.

Pure functions only: no filesystem writes, no network, no subprocesses. The
orchestration (which evals run, in what order, what the exit code is) lives in
``scripts/run_evals.py --gate``; this module decides *pass/fail* so it can be
unit-tested hermetically.

Failure contract:

* **Diffs** (only when both baseline and current carry the metric):
  retrieval MRR / hit@1 / hit@3 / hit@5, conversation ``pass_rate``, and the
  finetune ``tool_faithfulness`` must not drop. Skipped metrics (no LLM key,
  no judge) are ``None`` and are ignored — absence is not a regression.
* **Floors** (absolute, from PHASE3.md): the agent e2e check must exit 0; the
  finetune replay must read 1.0 (fake-LLM best case); exports must keep at
  least one training example when raw turns exist; generation faithfulness,
  when the judge produced values, must meet ``settings.faithfulness_threshold``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GATE_EPS = 1e-6

# (label, path) — metrics compared against the previous gate report.
COMPARE: list[tuple[str, tuple[str, ...]]] = [
    ("retrieval.mrr", ("retrieval", "mrr")),
    ("retrieval.hit_rate_at_1", ("retrieval", "hit_rate_at_1")),
    ("retrieval.hit_rate_at_3", ("retrieval", "hit_rate_at_3")),
    ("retrieval.hit_rate_at_5", ("retrieval", "hit_rate_at_5")),
    ("conversation.pass_rate", ("conversation", "pass_rate")),
    ("finetune.tool_faithfulness", ("finetune", "replay", "tool_faithfulness")),
]


def dig(report: dict, path: tuple[str, ...]) -> Any | None:
    cur: Any = report
    for key in path:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            return None
        cur = cur[key]
    return cur


def gate_regressions(prev: dict | None, current: dict) -> list[str]:
    if not prev:
        return []
    failures: list[str] = []
    for label, path in COMPARE:
        a = dig(prev, path)
        b = dig(current, path)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not isinstance(a, bool) and not isinstance(b, bool):
                if b < a - GATE_EPS:
                    failures.append(f"regression {label}: {a} -> {b}")
    return failures


def gate_floor_failures(current: dict, settings) -> list[str]:
    failures: list[str] = []

    agent = current.get("agent") or {}
    if agent.get("returncode", 0) != 0:
        failures.append(f"agent check failed (exit {agent.get('returncode')})")

    finetune = current.get("finetune") or {}
    if finetune.get("n_raw_turns", 0) > 0 and finetune.get("n_train", 0) < 1:
        failures.append("finetune train_examples < 1 (all exports were filtered out)")

    replay = finetune.get("replay") or {}
    if replay.get("error"):
        failures.append(f"finetune replay error: {replay['error']}")
    if not replay.get("skipped"):
        tf = replay.get("tool_faithfulness")
        if tf is not None and tf < 1.0 - GATE_EPS:
            failures.append(
                f"finetune tool_faithfulness {tf} < 1.0 (fake-LLM best-case bound)"
            )

    generation = current.get("generation") or {}
    if not generation.get("skipped"):
        mf = generation.get("mean_faithfulness")
        if mf is not None and mf < settings.faithfulness_threshold:
            failures.append(
                f"generation faithfulness {mf} < threshold {settings.faithfulness_threshold}"
            )

    return failures


def evaluate_gate(
    prev: dict | None, current: dict, settings
) -> tuple[bool, list[str]]:
    failures = gate_regressions(prev, current) + gate_floor_failures(current, settings)
    return (not failures), failures


def load_gate_report(path: str | Path) -> dict | None:
    """Return the previous gate report, or None if missing or unreadable.

    Callers must treat "file exists but returned None" as a failure: a corrupt
    baseline must not silently disable regression checking.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def write_gate_report(path: str | Path, report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
