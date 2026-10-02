"""Phase 2 SSE end-to-end check with a scripted fake LLM (hermetic).

Drives the real ``app.streaming.sse.sse_generator`` (which in turn runs the
agent streaming loop) and asserts the SSE event sequence contains the expected
``event:`` names. No network, no API key, no saved index required.

Usage:
    python scripts/e2e_sse_check.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fake_llm import build_fake_generator
from app.index import Index
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.streaming.sse import sse_generator
from app.tools.registry import build_tool_registry
from app.tracing import Trace


def parse_sse_events(chunks):
    """Parse SSE chunk strings -> list of (event_name, data) tuples."""
    events = []
    for chunk in chunks:
        event_name = "message"
        data_parts = []
        event_parts = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event_parts.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                data_parts.append(line.split(":", 1)[1].strip())
        if event_parts:
            event_name = ":".join(event_parts)
        data = "\n".join(data_parts)
        events.append((event_name, data))
    return events


async def main():
    settings = Settings()
    settings.use_rerank = False

    # Fake index: build a tiny in-memory index if none is saved.
    from app.embedding import get_embedder

    embedder = get_embedder(settings)
    index = Index(settings, embedder, None)
    if not index.load():
        print("No saved index; building from sources...")
        index.build_from_directory()
        index.save()

    # Isolated memory: e2e runs are synthetic and must not pollute the real
    # data/memory + data/episodic state.
    memory = MemoryManager(
        long_term_path=settings.trace_dir / "e2e_memory",
        episodic_path=settings.trace_dir / "e2e_memory",
    )
    memory.start_conversation()
    registry = build_tool_registry(index, memory, use_rerank=False)

    # Script: LLM first calls search_knowledge, then emits a final answer.
    script = [
        json.dumps(
            {
                "thought": "I should look up hybrid retrieval.",
                "tool": "search_knowledge",
                "args": {"query": "hybrid retrieval rrf"},
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "thought": "Now I can answer.",
                "tool": None,
                "args": None,
                "final_answer": "Hybrid retrieval combines BM25 and dense vectors with RRF [1].",
            }
        ),
    ]
    generator = build_fake_generator(script)

    trace = Trace(settings.trace_dir)
    chunks: list[str] = []
    async for chunk in sse_generator(
        "What is hybrid retrieval?",
        generator=generator,
        registry=registry,
        memory=memory,
        settings=settings,
        trace=trace,
        max_iterations=4,
        conv_id="sse_e2e_test",
    ):
        chunks.append(chunk)
    trace.save()

    events = parse_sse_events(chunks)
    print("=== SSE END-TO-END CHECK ===")
    print(f"chunks: {len(chunks)}")
    for event_name, data in events:
        preview = data if len(data) <= 80 else data[:77] + "..."
        print(f"  event: {event_name:<14} data: {preview}")

    seen = [name for name, _ in events]
    print("=== RESULT ===")
    print(f"event sequence: {seen}")
    required = {"tool_call", "tool_result", "final_answer", "close"}
    ok = required.issubset(seen)
    print(f"ok: {ok}")
    if not ok:
        raise SystemExit(1)
    print(f"trace_id: {trace.trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
