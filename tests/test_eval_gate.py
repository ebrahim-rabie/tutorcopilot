"""Hermetic gate for Phase 3 P3.2: the finetune eval runner + eval gate logic.

Fully offline (no LLM, no network, no saved index, no reranker): exercises

* ``FinetuneEvalRunner.data_metrics`` on a temp episodic dump (purity, dedupe,
  deterministic split, tool-call success rate);
* ``FinetuneEvalRunner.build_replay_script`` alignment contract (one tool
  decision per expected tool, then a final answer drawn from the exports);
* ``app.eval_gate`` diff/floor logic (regressions, absolute floors, skipped
  metrics ignored, first-run baseline) against synthetic reports;
* the fake-LLM generator reports itself available without an API key.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.eval_finetune import FinetuneEvalRunner
from app.eval_gate import (
    evaluate_gate,
    gate_floor_failures,
    gate_regressions,
    load_gate_report,
)
from app.fake_llm import build_fake_generator
from app.settings import Settings

TURN_LINES = [
    json.dumps(
        {
            "user_message": "What is hybrid retrieval?",
            "assistant_message": "Hybrid retrieval combines BM25 + dense vectors with RRF [1].",
            "tools_used": [{"name": "search_knowledge", "success": True}],
            "success": True,
        }
    ),
    json.dumps(
        {
            "user_message": "What is hybrid retrieval?",  # duplicate
            "assistant_message": "Hybrid retrieval combines BM25 + dense vectors with RRF [1].",
            "tools_used": [],
            "success": True,
        }
    ),
    json.dumps(
        {
            "user_message": "   ",  # blank -> purity drop
            "assistant_message": "dropped",
            "tools_used": [],
            "success": True,
        }
    ),
    json.dumps(
        {
            "user_message": "What is RRF?",
            "assistant_message": "RRF merges ranked lists.",
            "tools_used": [{"name": "search_knowledge", "success": False}],
            "success": False,
        }
    ),
]

CONV_TESTS = [
    json.dumps(
        {
            "id": "gt1",
            "name": "gate test",
            "turns": [
                {"user": "What is hybrid retrieval?", "expected_tools": ["search_knowledge"]},
                {"user": "And RRF?", "expected_tools": ["search_knowledge"]},
            ],
        }
    ),
    json.dumps(
        {
            "id": "gt2",
            "name": "no tool turn",
            "turns": [{"user": "Just answer", "expected_tools": []}],
        }
    ),
]


def _make_runner(tmp: Path) -> FinetuneEvalRunner:
    episodic = tmp / "episodic"
    episodic.mkdir(parents=True, exist_ok=True)
    (episodic / "c1.jsonl").write_text("\n".join(TURN_LINES) + "\n", encoding="utf-8")
    tests = tmp / "conversation_tests.jsonl"
    tests.write_text("\n".join(CONV_TESTS) + "\n", encoding="utf-8")
    settings = Settings(episodic_dir=episodic, trace_dir=tmp / "traces")
    return FinetuneEvalRunner(settings, test_path=tests)


def test_data_metrics_on_temp_dump():
    with tempfile.TemporaryDirectory() as tmp:
        metrics = _make_runner(Path(tmp)).data_metrics()

        assert metrics["n_raw_turns"] == 4
        assert metrics["n_pure_turns"] == 3
        assert metrics["n_examples"] == 2  # duplicate + blank collapsed away
        assert metrics["kept_ratio"] == 0.5
        assert metrics["dedup_ratio"] == round(1 - 2 / 3, 4)
        assert metrics["n_train"] + metrics["n_val"] == 2
        assert metrics["tool_calls_total"] == 2
        assert metrics["tool_calls_success_rate"] == 0.5


def test_replay_script_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        runner = _make_runner(Path(tmp))
        _, _, examples = runner._load_examples()
        generations = [e.generation for e in examples]
        tool_schemas = {
            "search_knowledge": {
                "required": ["query"],
                "parameters": {"properties": {"query": {"type": "string"}}},
            },
            "run_python": {
                "required": ["code"],
                "parameters": {"properties": {"code": {"type": "string"}}},
            },
        }
        script = runner.build_replay_script(generations, tool_schemas)

        # gt1: 2 turns x (1 tool + 1 final), gt2: 1 turn x (0 tool + 1 final)
        assert len(script) == 5
        steps = [json.loads(s) for s in script]

        tool_steps = [s for s in steps if s["tool"] is not None]
        final_steps = [s for s in steps if s["final_answer"] is not None]
        assert [s["tool"] for s in tool_steps] == ["search_knowledge", "search_knowledge"]
        assert tool_steps[0]["args"]["query"] == "What is hybrid retrieval?"
        assert len(final_steps) == 3
        # every final answer is an exported generation
        assert set(final_steps[i]["final_answer"] for i in range(3)) <= set(generations)


def test_replay_script_args_from_tool_schemas():
    """Required fields must be filled per the tool's schema — the loop drops
    tool calls whose args fail validation (e.g. run_python needs `code`)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = _make_runner(Path(tmp))
        tests = Path(tmp) / "conversation_tests.jsonl"
        tests.write_text(
            json.dumps(
                {
                    "id": "t",
                    "name": "two tools",
                    "turns": [
                        {"user": "Compute MRR", "expected_tools": ["run_python", "search_knowledge"]}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runner.test_path = tests
        tool_schemas = {
            "search_knowledge": {
                "required": ["query"],
                "parameters": {"properties": {"query": {"type": "string"}}},
            },
            "run_python": {
                "required": ["code"],
                "parameters": {"properties": {"code": {"type": "string"}}},
            },
        }
        script = runner.build_replay_script(["gen"], tool_schemas)
        steps = [json.loads(s) for s in script]
        assert len(steps) == 3  # 2 tool decisions + 1 final
        by_tool = {s["tool"]: s["args"] for s in steps if s["tool"]}
        assert by_tool["run_python"]["code"] == "print(1)"
        assert by_tool["search_knowledge"]["query"] == "Compute MRR"

        # no schema -> generic query fallback
        fallback = [json.loads(s) for s in runner.build_replay_script(["gen"])]
        assert fallback[0]["tool"] == "run_python"
        assert fallback[0]["args"] == {"query": "Compute MRR"}


def test_replay_skips_without_index():
    with tempfile.TemporaryDirectory() as tmp:
        import asyncio

        runner = _make_runner(Path(tmp))
        replay = asyncio.run(runner.run_replay(None))
        assert replay["skipped"] is True
        assert "no index" in replay["reason"]


def test_fake_generator_available_without_api_key():
    gen = build_fake_generator([json.dumps({"final_answer": "x", "tool": None, "args": None})])
    assert gen.available() is True


def _base_current(**overrides) -> dict:
    current = {
        "retrieval": {"mrr": 0.8, "hit_rate_at_1": 0.7, "hit_rate_at_3": 0.9, "hit_rate_at_5": 1.0},
        "generation": {"skipped": True, "mean_faithfulness": None},
        "conversation": {"skipped": True, "n_tests": 0, "pass_rate": None},
        "agent": {"returncode": 0},
        "finetune": {
            "n_raw_turns": 10,
            "n_train": 8,
            "n_val": 2,
            "replay": {"skipped": False, "tool_faithfulness": 1.0},
        },
    }
    current.update(overrides)
    return current


def test_gate_first_run_and_stable_metrics_pass():
    settings = Settings()
    # no baseline -> only floors apply
    ok, failures = evaluate_gate(None, _base_current(), settings)
    assert ok and failures == []
    # identical baseline -> no regression
    cur = _base_current()
    ok, failures = evaluate_gate(cur, cur, settings)
    assert ok and failures == []


def test_gate_detects_regressions():
    prev = _base_current()
    cur = _base_current(
        retrieval={"mrr": 0.7, "hit_rate_at_1": 0.7, "hit_rate_at_3": 0.9, "hit_rate_at_5": 1.0},
        conversation={"skipped": False, "n_tests": 3, "pass_rate": 0.6667},
        finetune={"n_raw_turns": 10, "n_train": 8, "n_val": 2, "replay": {"skipped": False, "tool_faithfulness": 0.75}},
    )
    # conversation pass_rate 1.0 -> ... prev had None (skipped): no diff there
    prev["conversation"] = {"skipped": False, "n_tests": 3, "pass_rate": 1.0}
    failures = gate_regressions(prev, cur)
    labels = " | ".join(failures)
    assert "retrieval.mrr: 0.8 -> 0.7" in labels
    assert "conversation.pass_rate: 1.0 -> 0.6667" in labels
    assert "finetune.tool_faithfulness: 1.0 -> 0.75" in labels
    assert "hit_rate_at_1" not in labels  # unchanged metric not reported


def test_gate_ignores_skipped_metrics():
    prev = _base_current()
    cur = _base_current()  # generation/conversation both skipped -> None paths
    assert gate_regressions(prev, cur) == []


def test_gate_floors():
    settings = Settings()

    agent_down = _base_current(agent={"returncode": 1})
    assert any("agent check" in f for f in gate_floor_failures(agent_down, settings))

    all_filtered = _base_current(
        finetune={"n_raw_turns": 10, "n_train": 0, "n_val": 0, "replay": {"skipped": True}}
    )
    assert any("train_examples" in f for f in gate_floor_failures(all_filtered, settings))

    bad_replay = _base_current(
        finetune={"n_raw_turns": 10, "n_train": 8, "n_val": 2, "replay": {"skipped": False, "tool_faithfulness": 0.5}}
    )
    assert any("tool_faithfulness" in f for f in gate_floor_failures(bad_replay, settings))

    low_faith = _base_current(generation={"skipped": False, "mean_faithfulness": 2})
    assert any("threshold" in f for f in gate_floor_failures(low_faith, settings))

    # skipped generation (hermetic, no judge) must NOT trip the faithfulness floor
    assert gate_floor_failures(_base_current(), settings) == []


def test_load_gate_report_handles_missing_and_corrupt(tmp_path: Path):
    assert load_gate_report(tmp_path / "nope.json") is None
    corrupt = tmp_path / "gate_latest.json"
    corrupt.write_text("{not json", encoding="utf-8")
    # exists-but-None is the caller's signal to FAIL, not to skip diffing
    assert load_gate_report(corrupt) is None
    good = tmp_path / "good.json"
    good.write_text('{"retrieval": {"mrr": 0.5}}', encoding="utf-8")
    assert load_gate_report(good) == {"retrieval": {"mrr": 0.5}}
