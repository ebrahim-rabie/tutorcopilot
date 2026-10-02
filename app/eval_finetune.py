"""Phase 3 (P3.2): eval over the exported LoRA training data.

``FinetuneEvalRunner`` answers two questions before any fine-tune starts:

1. **Is the export still healthy?** Pure, hermetic metrics over the episodic
   dumps: how many raw turns survive the purity gate, how much dedupe removed,
   the train/val split sizes, and the tool-call success rate of what survived.
2. **Is agent tool behaviour still compatible with the data?** A
   *tool_faithfulness* replay: the existing ``ConversationEvalHarness`` runs the
   real agent loop and real tool registry, but with a scripted fake LLM that
   calls each expected tool in order and answers with an exported generation.
   That is the hermetic best-case bound — it can only read ``1.0`` while the
   plumbing (registry, loop, JSON decision format) still works; anything lower
   means the tool behaviour the training data assumes has regressed.

The replay needs an ``Index``; the gate passes the one Phase 1 already built.
Without one (pure metric runs, unit tests) the replay reports ``skipped``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from app.eval_conversation import ConversationEvalHarness, load_conversation_tests
from app.finetune.check import SUSPICIOUS_PREFIX
from app.finetune.schema import dedupe, load_episodic_turns, split, to_examples
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.tools.registry import build_tool_registry

# Safe stand-in values per required field name (replay executes tools for real,
# so run_python gets a trivial snippet and fetch_url a fail-fast local address).
_FIELD_VALUES = {
    "code": "print(1)",
    "url": "http://127.0.0.1:9",
    "title": "replay",
    "content": "replay",
}

_TYPE_FALLBACKS = {
    "string": "",
    "integer": 4,
    "number": 0.0,
    "boolean": False,
    "array": [],
    "object": {},
}


def _tool_args(tool: str, user: str, tool_schemas: dict[str, dict] | None) -> dict:
    """Build args that pass ``validate_args`` for ``tool``.

    Known field names win (query/code/url/...); anything else falls back by
    schema type, then to the user message for strings. Unknown tools (no
    schema) get the generic ``{"query": user}``.
    """
    schema = (tool_schemas or {}).get(tool)
    if schema is None:
        return {"query": user}
    props = (schema.get("parameters") or {}).get("properties") or {}
    args: dict = {}
    for field_name in schema.get("required") or []:
        if field_name in _FIELD_VALUES:
            args[field_name] = _FIELD_VALUES[field_name]
        elif field_name == "query" or field_name.endswith("_query"):
            args[field_name] = user
        else:
            ftype = (props.get(field_name) or {}).get("type", "string")
            if ftype == "string":
                args[field_name] = user
            else:
                args[field_name] = _TYPE_FALLBACKS.get(ftype, user)
    return args


class FinetuneEvalRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        test_path: str | Path | None = None,
    ):
        self.settings = settings or Settings()
        self.test_path = Path(test_path or "evals/conversation_tests.jsonl")

    def _load_examples(self):
        turns = load_episodic_turns(self.settings.episodic_dir)
        pure = to_examples(turns)
        examples = dedupe(pure)
        return turns, pure, examples

    def data_metrics(self) -> dict[str, Any]:
        """Pure export health metrics (no network, no index, no LLM)."""
        turns, pure, examples = self._load_examples()
        train, val = split(examples)
        tc_total = sum(len(e.tool_calls) for e in examples)
        tc_ok = sum(1 for e in examples for tc in e.tool_calls if tc.success)
        n_failed = sum(
            1 for e in examples if e.generation.startswith(SUSPICIOUS_PREFIX)
        )
        n_turns = len(turns)
        n_raw = len(pure)
        n_kept = len(examples)
        return {
            "n_raw_turns": n_turns,
            "n_pure_turns": n_raw,
            "n_examples": n_kept,
            "kept_ratio": round(n_kept / n_turns, 4) if n_turns else 0.0,
            "dedup_ratio": round(1 - n_kept / n_raw, 4) if n_raw else 0.0,
            "n_train": len(train),
            "n_val": len(val),
            "tool_calls_total": tc_total,
            "tool_calls_success_rate": round(tc_ok / tc_total, 4) if tc_total else None,
            "n_failed_generations": n_failed,
            "failed_generation_ratio": round(n_failed / n_kept, 4) if n_kept else 0.0,
        }

    def build_replay_script(
        self,
        generations: list[str],
        tool_schemas: dict[str, dict] | None = None,
    ) -> list[str]:
        """Script the fake LLM: one tool decision per expected tool, then a
        final answer drawn (cycled) from the exported generations.

        Tool args are built from the tool's registered schema — the agent loop
        only records a tool call when ``validate_args`` passes, so every
        ``required`` field must be present (e.g. ``run_python`` needs ``code``,
        not ``query``). Without a schema the loop's generic ``{"query": user}``
        fallback is used.

        Length must equal the agent loop's LLM call count per turn
        (len(expected_tools) + 1) or the replay fails loudly.
        """
        tests = load_conversation_tests(self.test_path)
        answers = itertools.cycle(generations) if generations else None
        script: list[str] = []
        for test in tests:
            for turn in test.turns:
                for tool in turn.get("expected_tools") or []:
                    script.append(
                        json.dumps(
                            {
                                "thought": f"Replay: call {tool} as recorded.",
                                "tool": tool,
                                "args": _tool_args(tool, turn["user"], tool_schemas),
                                "final_answer": None,
                            }
                        )
                    )
                answer = next(answers) if answers is not None else "(no exported generations)"
                script.append(
                    json.dumps(
                        {
                            "thought": "Replay: answer with exported generation.",
                            "tool": None,
                            "args": None,
                            "final_answer": answer,
                        }
                    )
                )
        return script

    async def run_replay(self, index, use_rerank: bool = False) -> dict[str, Any]:
        """Replay conversation tests through the real agent loop with the fake
        LLM. Returns turn-level ``tool_faithfulness`` plus test-level
        ``pass_rate``; ``skipped``/``error`` describe why it did not run.
        """
        if index is None:
            return {"skipped": True, "reason": "no index provided"}
        if not self.test_path.exists():
            return {"skipped": True, "reason": f"missing {self.test_path}"}
        _, _, examples = self._load_examples()
        if not examples:
            return {"skipped": True, "reason": "no exported training examples"}

        self.settings.use_rerank = use_rerank
        memory = MemoryManager(
            long_term_path=self.settings.trace_dir / "eval_memory",
            episodic_path=self.settings.trace_dir / "eval_memory",
        )
        registry = build_tool_registry(index, memory, use_rerank)
        tool_schemas = {
            name: {
                "required": registry.get(name).required,
                "parameters": registry.get(name).parameters,
            }
            for name in registry.list_names()
            if registry.get(name) is not None
        }
        script = self.build_replay_script(
            [e.generation for e in examples], tool_schemas
        )
        from app.fake_llm import build_fake_generator

        generator = build_fake_generator(script, self.settings)
        harness = ConversationEvalHarness(self.settings, generator, registry, memory)
        try:
            report = await harness.run(self.test_path)
        except Exception as exc:  # script misalignment or tool/loop breakage
            return {"skipped": False, "error": f"{type(exc).__name__}: {exc}"}

        results = report.get("results") or []
        n_turns = sum(r["n_turns"] for r in results)
        n_match = sum(
            1 for r in results for tr in r["turn_results"] if tr["tools_match"]
        )
        n_pass = sum(1 for r in results if r["all_tools_match"])
        return {
            "skipped": False,
            "n_tests": len(results),
            "n_turns": n_turns,
            "n_tools_match": n_match,
            "tool_faithfulness": round(n_match / n_turns, 4) if n_turns else None,
            "pass_rate": round(n_pass / len(results), 4) if results else None,
        }
