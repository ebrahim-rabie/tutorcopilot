from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.agent.loop import run_agent_loop
from app.generate import Generator, retry_transient
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.tools.registry import ToolRegistry
from app.tracing import Trace


@dataclass
class ConversationTestCase:
    """A multi-turn conversation test case."""

    id: str
    name: str
    turns: list[
        dict
    ]  # Each: {"user": "...", "expected_tools": [...], "expected_topics": [...], "min_faithfulness": int}
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict) -> ConversationTestCase:
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            turns=data["turns"],
            difficulty=str(data.get("difficulty", "medium")),
        )


def load_conversation_tests(path: str | Path) -> list[ConversationTestCase]:
    tests = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            tests.append(ConversationTestCase.from_dict(json.loads(line)))
    return tests


async def run_conversation_eval(
    test: ConversationTestCase,
    generator: Generator,
    registry: ToolRegistry,
    memory: MemoryManager,
    settings: Settings,
    judge=None,
) -> dict:
    """Run a multi-turn conversation test case."""
    conv_id = f"eval_{test.id}"
    memory.start_conversation(conv_id)
    trace = Trace(settings.trace_dir)

    turn_results = []
    total_faithfulness = 0
    faithful_count = 0

    for i, turn in enumerate(test.turns):
        user_msg = turn["user"]
        expected_tools = turn.get("expected_tools", [])
        min_faithfulness = turn.get("min_faithfulness", 3)

        result = await run_agent_loop(
            question=user_msg,
            generator=generator,
            registry=registry,
            memory=memory,
            settings=settings,
            trace=trace,
            max_iterations=settings.agent_max_iterations,
            conv_id=conv_id,
        )

        # Check tool usage
        tools_match = set(result.tools_called) >= set(expected_tools)

        # Faithfulness check (if judge available)
        faithfulness = None
        if judge and judge.available() and result.steps:
            # Get context from last retrieval
            context = ""
            for step in result.steps:
                if (
                    step.tool_name == "search_knowledge"
                    and step.tool_result
                    and step.tool_result.success
                ):
                    context = "\n".join([r.get("text", "") for r in step.tool_result.data or []])
                    break
            if context:
                verdict = retry_transient(
                    lambda u=user_msg, r=result, c=context: judge.rate_grounded(u, r.answer, c)
                )
                faithfulness = verdict.get("faithfulness")
                if faithfulness is not None:
                    total_faithfulness += faithfulness
                    faithful_count += 1

        turn_results.append(
            {
                "turn": i + 1,
                "user": user_msg,
                "answer": result.answer,
                "tools_called": result.tools_called,
                "tools_match": tools_match,
                "expected_tools": expected_tools,
                "faithfulness": faithfulness,
                "min_faithfulness": min_faithfulness,
                "success": result.success,
            }
        )

    memory.end_conversation({"test_id": test.id}, persist=False)
    trace.save()

    avg_faithfulness = total_faithfulness / faithful_count if faithful_count else None
    all_tools_match = all(r["tools_match"] for r in turn_results)

    return {
        "test_id": test.id,
        "name": test.name,
        "difficulty": test.difficulty,
        "n_turns": len(test.turns),
        "all_tools_match": all_tools_match,
        "avg_faithfulness": avg_faithfulness,
        "turn_results": turn_results,
    }


class ConversationEvalHarness:
    def __init__(
        self,
        settings: Settings,
        generator: Generator,
        registry: ToolRegistry,
        memory: MemoryManager,
    ):
        self.settings = settings
        self.generator = generator
        self.registry = registry
        self.memory = memory

    async def run(self, test_path: str | Path) -> dict:
        if not self.generator.available():
            return {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "n_tests": 0,
                "skipped": True,
                "reason": "LLM not configured (set TC_LLM_API_KEY)",
            }
        tests = load_conversation_tests(test_path)
        start = time.time()
        results = []
        for test in tests:
            result = await run_conversation_eval(
                test, self.generator, self.registry, self.memory, self.settings
            )
            results.append(result)

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_tests": len(tests),
            "results": results,
            "duration_sec": round(time.time() - start, 2),
        }
