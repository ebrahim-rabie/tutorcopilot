"""End-to-end verification of the Stage 2 agent loop with a mocked LLM client."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import run_agent_loop
from app.fake_llm import build_fake_generator
from app.index import Index
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.tools.registry import build_tool_registry
from app.tracing import Trace


def main():
    settings = Settings()
    settings.use_rerank = False

    # Isolated memory: e2e runs are synthetic and must not pollute the real
    # data/memory + data/episodic state.
    memory = MemoryManager(
        long_term_path=settings.trace_dir / "e2e_memory",
        episodic_path=settings.trace_dir / "e2e_memory",
    )
    memory.start_conversation()

    # Use a null index via a minimal dummy, OR build the real one.
    # For a full end-to-end we build the real index (needs saved index present).
    from app.embedding import get_embedder

    embedder = get_embedder(settings)
    index = Index(settings, embedder, None)
    if not index.load():
        print("No saved index; building from sources...")
        index.build_from_directory()
        index.save()

    registry = build_tool_registry(index, memory, use_rerank=False)

    # Script: LLM first calls search_knowledge, then produces a final answer.
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
    result = asyncio.run(
        run_agent_loop(
            question="What is hybrid retrieval?",
            generator=generator,
            registry=registry,
            memory=memory,
            settings=settings,
            trace=trace,
            max_iterations=4,
            conv_id="e2e_test",
        )
    )
    trace.save()

    print("=== END-TO-END AGENT TEST ===")
    print(f"success: {result.success}")
    print(f"answer: {result.answer}")
    print(f"tools_called: {result.tools_called}")
    print(f"n_steps: {len(result.steps)}")
    print(f"latency_ms: {result.total_latency_ms:.1f}")
    print(f"tokens: {result.tokens_used}")
    print(f"trace_id: {result.trace_id}")
    for s in result.steps:
        print(
            f"  step {s.iteration}: tool={s.tool_name} ok={s.tool_result.success if s.tool_result else None}"
        )
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
