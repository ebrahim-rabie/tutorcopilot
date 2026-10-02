"""Phase 3 (P3.1): training-example schema lifted from an agent trace.

A ``TrainingExample`` captures one labeled (question -> assistant generation)
pair ready for LoRA fine-tuning, plus the tool calls made along the way so an
eval gate can replay the trace and confirm the fine-tune didn't regress tool use.

Traces follow the schema in ``app/tracing.py``:
``{"trace_id", "started_at", "events": [{"ts", "name", "data"}]}`` where the
agent events are named ``tool_call`` / ``tool_result`` / ``final_answer``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TrainingExampleSplit = tuple[list["TrainingExample"], list["TrainingExample"]]


@dataclass
class ToolCallRecord:
    """One tool call observed inside a trace event stream."""

    name: str
    args: dict[str, Any]
    success: bool
    error: str | None = None


@dataclass
class TrainingExample:
    """A single validated (question -> generation) pair for LoRA fine-tuning.

    Attributes:
        question: the user question (input).
        generation: the assistant's final answer / thought trace (target).
        tool_calls: every tool the agent invoked to produce the generation.
        success: whether the underlying trace recorded the answer as successful
            (True => great positive; False => hard-negative / to-be-curated).
        trace_id / conv_id: provenance so the eval gate can replay + regress.
    """

    question: str
    generation: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    success: bool = True
    trace_id: str | None = None
    conv_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "generation": self.generation,
            "tool_calls": [
                {
                    "name": tc.name,
                    "args": tc.args,
                    "success": tc.success,
                    "error": tc.error,
                }
                for tc in self.tool_calls
            ],
            "success": self.success,
            "trace_id": self.trace_id,
            "conv_id": self.conv_id,
        }

    def is_valid(self) -> bool:
        """Hermetic purity gate: non-empty I/O is all we require."""
        return bool(self.question.strip()) and bool(self.generation.strip())

    @classmethod
    def from_dict(cls, data: dict) -> TrainingExample:
        """Strict inverse of ``to_dict`` — raises ValueError on schema drift.

        Used by P3.3's ``--check-only`` to assert exports are trainable before
        any (large) ML dependency is imported.
        """
        if not isinstance(data, dict):
            raise ValueError("example must be a JSON object")
        question = data.get("question")
        generation = data.get("generation")
        if not isinstance(question, str) or not isinstance(generation, str):
            raise ValueError("question/generation must be strings")

        raw_calls = data.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls must be a list")
        tool_calls: list[ToolCallRecord] = []
        for i, tc in enumerate(raw_calls):
            if not isinstance(tc, dict):
                raise ValueError(f"tool_calls[{i}] must be an object")
            name = tc.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"tool_calls[{i}].name must be a non-empty string")
            args = tc.get("args", {})
            if not isinstance(args, dict):
                raise ValueError(f"tool_calls[{i}].args must be an object")
            if "success" in tc and not isinstance(tc["success"], bool):
                raise ValueError(f"tool_calls[{i}].success must be a boolean")
            tool_calls.append(
                ToolCallRecord(
                    name=name,
                    args=args,
                    success=bool(tc.get("success", True)),
                    error=tc.get("error"),
                )
            )

        if "success" in data and not isinstance(data["success"], bool):
            raise ValueError("success must be a boolean")
        return cls(
            question=question,
            generation=generation,
            tool_calls=tool_calls,
            success=bool(data.get("success", True)),
            trace_id=data.get("trace_id"),
            conv_id=data.get("conv_id"),
        )


def load_episodic_turns(episodic_dir: str | Path) -> list[dict]:
    """Yield every turn across all ``*.json``/``*.jsonl`` episodic dumps.

    Episodic dumps are one document per conversation:
    ``{"conversation_id", "created_at", "metadata", "turns": [{...}]}``.
    A ``conversation_id`` found at the document level is merged into every
    turn so ``to_examples`` can attach it for provenance. JSONL dumps
    (one turn per line) also work.

    Synthetic dumps never train: conversation-eval documents (``eval_*`` /
    ``metadata.test_id``) are skipped even if they end up in the directory —
    their generations come from a fake LLM, not real conversations.
    """
    turns: list[dict] = []

    def _synthetic(conv_id, metadata: dict | None = None) -> bool:
        if conv_id is not None and str(conv_id).startswith("eval_"):
            return True
        return bool((metadata or {}).get("test_id"))

    for dump in sorted(Path(episodic_dir).glob("*.json*")):
        try:
            lines = [
                ln.strip()
                for ln in dump.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        except OSError:
            continue
        if not lines:
            continue
        if dump.suffix.lower() == ".jsonl":
            for line in lines:
                try:
                    turn = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(turn, dict) or _synthetic(
                    turn.get("conversation_id") or turn.get("conv_id"), turn.get("metadata")
                ):
                    continue
                turns.append(turn)
            continue
        # one conversation document per .json file
        try:
            doc = json.loads("\n".join(lines))
        except json.JSONDecodeError:
            continue
        conv_id = doc.get("conversation_id") or doc.get("conv_id")
        if _synthetic(conv_id, doc.get("metadata")):
            continue
        for t in doc.get("turns") or []:
            if not isinstance(t, dict):
                continue
            if conv_id and not t.get("conversation_id"):
                t["conversation_id"] = conv_id
            turns.append(t)
    return turns


def to_examples(turns: list[dict]) -> list[TrainingExample]:
    """Pure hermetic gate: keep only non-empty question + generation.

    Also lifts any ``tools_used``/``tools`` list on the turn into
    ``ToolCallRecord`` entries so tool-use survives into the LoRA data.
    """
    out: list[TrainingExample] = []
    for t in turns:
        question = (t.get("user_message") or "").strip()
        generation = (t.get("assistant_message") or "").strip()
        if not question or not generation:
            continue

        tool_calls: list[ToolCallRecord] = []
        for tool in t.get("tools_used") or t.get("tools") or []:
            if isinstance(tool, str):
                tool_calls.append(
                    ToolCallRecord(
                        name=tool,
                        args={},
                        success=True,
                        error=None,
                    )
                )
            elif isinstance(tool, dict):
                name = str(tool.get("name") or tool.get("tool") or "?")
                args = tool.get("args") or tool.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                success = bool(tool.get("success", True))
                error = tool.get("error")
                tool_calls.append(
                    ToolCallRecord(
                        name=name,
                        args=args,
                        success=success,
                        error=error,
                    )
                )

        out.append(
            TrainingExample(
                question=question,
                generation=generation,
                tool_calls=tool_calls,
                success=bool(t.get("success", True)),
                trace_id=str(t.get("trace_id")) if t.get("trace_id") else None,
                conv_id=str(t.get("conversation_id") or t.get("conv_id"))
                if t.get("conversation_id") or t.get("conv_id")
                else None,
            )
        )
    return out


def dedupe(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Drop exact (question, generation) duplicates, first occurrence wins."""
    seen: set[tuple[str, str]] = set()
    out: list[TrainingExample] = []
    for e in examples:
        key = (e.question, e.generation)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def split(
    examples: list[TrainingExample], seed: int = 42, val_ratio: float = 0.2
) -> TrainingExampleSplit:
    """Deterministic seeded split into (train, val)."""
    pool = list(examples)
    random.Random(seed).shuffle(pool)
    # int() truncation must never strand the only example(s) in val:
    # train gets at least 1 whenever any examples exist.
    cut = max(1, int(len(pool) * (1 - val_ratio))) if pool else 0
    return pool[:cut], pool[cut:]
