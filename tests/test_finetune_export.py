"""Hermetic gate for Phase 3 P3.1+P3.2: the LoRA data export pipeline.

Runs entirely offline (no LLM, no network, no saved index, no finetune
dependencies): builds a tiny episodic JSONL in a temp dir, runs the exporter's
pure functions (``load_episodic_turns`` / ``to_examples`` / ``dedupe`` /
``split``) and asserts the training-schema purity contract:

* non-empty question + generation survive;
* exact question+generation duplicates collapse to one;
* the seeded 80/20 split is deterministic (same seed -> same cut);
* every emitted TrainingExample is ``is_valid()``.

Importantly this test imports from ``app.finetune.schema`` only — the package
must stay importable without torch/transformers/peft/trl installed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.finetune.schema import (
    dedupe,
    load_episodic_turns,
    split,
    to_examples,
)

# A tiny episodic dump: 1 duplicate + 1 empty-question (must be dropped).
TURN_LINES = [
    json.dumps(
        {
            "user_message": "What is hybrid retrieval?",
            "assistant_message": "Hybrid retrieval combines BM25 + dense vectors with RRF [1].",
            "tools_used": [{"name": "search_knowledge", "success": True}],
            "success": True,
            "trace_id": "t1",
        }
    ),
    json.dumps(
        {
            "user_message": "What is hybrid retrieval?",  # duplicate
            "assistant_message": "Hybrid retrieval combines BM25 + dense vectors with RRF [1].",
            "tools_used": [{"name": "search_knowledge", "success": True}],
            "success": True,
            "trace_id": "t1b",
        }
    ),
    json.dumps(
        {
            "user_message": "   ",
            "assistant_message": "blank question -> drop",
            "tools_used": [],
            "success": True,
            "trace_id": "t2",
        }
    ),
    json.dumps(
        {
            "user_message": "What is RRF?",
            "assistant_message": "RRF (Reciprocal Rank Fusion) merges ranked lists.",
            "tools_used": [{"name": "search_knowledge", "success": True}],
            "success": False,  # hard-negative; exporter keeps but marks success=False
            "trace_id": "t3",
        }
    ),
]


def _write_dump(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    f = d / "episodic_c1.jsonl"
    f.write_text("\n".join(TURN_LINES) + "\n", encoding="utf-8")
    return f


def test_exporter_pipeline_hermetic():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "episodic"
        _write_dump(d)

        turns = load_episodic_turns(d)
        examples = dedupe(to_examples(turns))

        # purity: blank-question dropped, duplicate collapsed
        assert len(examples) == 2
        questions = [e.question for e in examples]
        assert "What is hybrid retrieval?" in questions  # once after dedupe
        assert questions.count("What is hybrid retrieval?") == 1
        assert "What is RRF?" in questions

        # hermetic validity gate
        assert all(e.is_valid() for e in examples)

        # deterministic seeded 80/20 split
        train_a, val_a = split(examples, seed=42, val_ratio=0.2)
        train_b, val_b = split(examples, seed=42, val_ratio=0.2)
        assert [e.question for e in train_a] == [e.question for e in train_b]
        assert [e.question for e in val_a] == [e.question for e in val_b]
        n = len(examples)
        assert len(train_a) == int(n * (1 - 0.2))  # val_ratio=0.2 -> 20% val
        assert len(val_a) == n - len(train_a)

        # hard-negative survives with success=False (curation hook)
        rrf = next(e for e in examples if e.question == "What is RRF?")
        assert rrf.success is False


def test_split_never_strands_lone_example_in_val():
    from app.finetune.schema import TrainingExample, split

    ex = TrainingExample(question="What is RRF?", generation="Reciprocal rank fusion.")
    train, val = split([ex], seed=42, val_ratio=0.2)
    assert len(train) == 1
    assert len(val) == 0  # check_exports then fails honestly on empty val

    train2, val2 = split([ex, ex], seed=42, val_ratio=0.2)
    assert len(train2) == 1
    assert len(val2) == 1


def test_synthetic_eval_dumps_never_train():
    """eval_* / metadata.test_id dumps are skipped even if present on disk.

    Conversation-eval runs answer with a fake LLM; their dumps must never
    enter the LoRA export (they once polluted data/episodic wholesale).
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "episodic"
        d.mkdir(parents=True, exist_ok=True)

        # real conversation document -> kept
        (d / "real_conv.json").write_text(
            json.dumps(
                {
                    "conversation_id": "abc123",
                    "turns": [
                        {"user_message": "q real", "assistant_message": "a real", "tools_used": []}
                    ],
                }
            ),
            encoding="utf-8",
        )
        # conversation-eval dump (eval_ id + test_id metadata) -> skipped
        (d / "eval_conv1.json").write_text(
            json.dumps(
                {
                    "conversation_id": "eval_conv1",
                    "metadata": {"test_id": "conv1"},
                    "turns": [
                        {"user_message": "q eval", "assistant_message": "hello", "tools_used": []}
                    ],
                }
            ),
            encoding="utf-8",
        )
        # eval_ conversation id without metadata -> still skipped
        (d / "eval_other.json").write_text(
            json.dumps(
                {
                    "conversation_id": "eval_other",
                    "turns": [
                        {"user_message": "q eval2", "assistant_message": "hello", "tools_used": []}
                    ],
                }
            ),
            encoding="utf-8",
        )
        # jsonl line carrying an eval_ conversation id -> skipped
        (d / "mixed.jsonl").write_text(
            json.dumps(
                {
                    "conversation_id": "eval_conv1",
                    "user_message": "q eval3",
                    "assistant_message": "hello",
                }
            )
            + "\n"
            + json.dumps(
                {"conversation_id": "z9", "user_message": "q jsonl", "assistant_message": "a jsonl"}
            )
            + "\n",
            encoding="utf-8",
        )

        turns = load_episodic_turns(d)
        users = [t.get("user_message") for t in turns]
        assert "q real" in users
        assert "q jsonl" in users
        assert not any(u and u.startswith("q eval") for u in users)


def test_schema_importable_without_ml_deps():
    """The finetune package must not import torch/transformers at import time."""
    import app.finetune  # noqa: F401  (import side effect is the check)
    import app.finetune.schema as schema_mod

    assert hasattr(schema_mod, "TrainingExample")
