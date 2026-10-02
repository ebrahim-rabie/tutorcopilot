import asyncio
import json

from app.agent.loop import (
    _extract_json,
    _parse_decision,
    _plan,
    build_system_prompt,
    run_agent_loop,
    run_agent_stream,
)
from app.generate import Generator
from app.index import Index
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.tools.registry import build_tool_registry
from app.tracing import Trace
from tests.helpers import FakeCompletion, FakeOpenAIClient, HashEmbedder


def make_env(tmp_path):
    """Build a settings object pointing at tmp_path (no saved index writes)."""
    settings = Settings()
    settings.use_rerank = False
    settings.trace_dir = tmp_path / "traces"
    settings.index_dir = tmp_path / "index"
    return settings


def build_test_registry(tmp_path):
    settings = make_env(tmp_path)
    embedder = HashEmbedder()
    index = Index(settings, embedder, None)
    index.add_documents(
        {
            "hybrid-retrieval.md": "# Hybrid Retrieval\nHybrid retrieval combines BM25 sparse scoring with dense vector search, fused with Reciprocal Rank Fusion (RRF).\nRRF typically uses k=60.",
        }
    )
    memory = MemoryManager(
        long_term_path=tmp_path / "memory",
        episodic_path=tmp_path / "episodic",
    )
    memory.start_conversation()
    registry = build_tool_registry(index, memory, use_rerank=False)
    return settings, index, memory, registry


def run_loop(
    generator,
    settings,
    registry,
    memory,
    trace,
    question="What is hybrid retrieval?",
    max_iterations=5,
):
    return asyncio.run(
        run_agent_loop(
            question=question,
            generator=generator,
            registry=registry,
            memory=memory,
            settings=settings,
            trace=trace,
            max_iterations=max_iterations,
            conv_id="test_conv",
        )
    )


def test_agent_loop_success(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    script = [
        json.dumps(
            {
                "thought": "look up",
                "tool": "search_knowledge",
                "args": {"query": "hybrid retrieval"},
                "final_answer": None,
            }
        ),
        json.dumps(
            {
                "thought": "done",
                "tool": None,
                "args": None,
                "final_answer": "BM25 + dense + RRF [1].",
            }
        ),
    ]
    client = FakeOpenAIClient(script)
    gen = Generator(settings, client=client)

    result = run_loop(gen, settings, registry, memory, trace)

    assert result.success
    assert result.answer == "BM25 + dense + RRF [1]."
    assert result.tools_called == ["search_knowledge"]
    assert len(result.steps) == 2


def test_agent_loop_unknown_tool(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    final = json.dumps(
        {"thought": "recover", "tool": None, "args": None, "final_answer": "Recovered."}
    )
    script = [
        json.dumps(
            {"thought": "bad tool", "tool": "nonexistent_tool", "args": {}, "final_answer": None}
        ),
        final,  # grounding forces a search before this recovery is accepted
        final,
    ]
    client = FakeOpenAIClient(script)
    gen = Generator(settings, client=client)

    result = run_loop(gen, settings, registry, memory, trace)

    assert result.success
    assert result.answer == "Recovered."
    assert result.steps[0].tool_result.success is False
    assert "unknown" in (result.steps[0].tool_result.error or "").lower()
    assert result.tools_called == ["search_knowledge"]


def test_agent_loop_cycle_detection(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    always = json.dumps(
        {
            "thought": "repeat",
            "tool": "search_knowledge",
            "args": {"query": "same"},
            "final_answer": None,
        }
    )
    client = FakeOpenAIClient(always=always)
    gen = Generator(settings, client=client)

    result = run_loop(gen, settings, registry, memory, trace, max_iterations=10)

    # Cycle detection should stop it before 10 iterations, with a graceful message.
    assert len(result.steps) < 10
    assert result.success is True
    assert "repeating" in result.answer  # fallback message


def test_agent_loop_saves_memory(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    always = json.dumps({"thought": "done", "tool": None, "args": None, "final_answer": "Answer."})
    client = FakeOpenAIClient(always=always)
    gen = Generator(settings, client=client)

    run_loop(gen, settings, registry, memory, trace)

    assert len(memory.short_term.turns) == 1
    assert memory.short_term.turns[0].user_message == "What is hybrid retrieval?"
    assert memory.short_term.turns[0].assistant_message == "Answer."


def test_agent_loop_gets_context_respects_memory(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    memory.add_turn("Q old", "A old", ["search_knowledge"], success=True, trace_id="t0")
    trace = Trace(settings.trace_dir)
    always = json.dumps({"thought": "done", "tool": None, "args": None, "final_answer": "Answer."})
    client = FakeOpenAIClient(always=always)
    gen = Generator(settings, client=client)

    run_loop(gen, settings, registry, memory, trace, question="New question?")

    # The second user prompt should contain the prior conversation context.
    last_user_prompts = client.chat.completions.calls
    assert any("Q old" in str(p) for p in last_user_prompts for m in p if m.get("role") == "user")


class _FailingCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs.get("messages", []))
        raise RuntimeError("simulated API rejection")


class FailingOpenAIClient:
    def __init__(self):
        from types import SimpleNamespace

        self.chat = SimpleNamespace(completions=_FailingCompletions())


async def _collect(agen):
    return [event async for event in agen]


def test_agent_loop_llm_error_degrades(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    client = FailingOpenAIClient()
    gen = Generator(settings, client=client)

    result = run_loop(gen, settings, registry, memory, trace)

    assert result.success is False
    assert "LLM call failed" in result.answer
    assert len(client.chat.completions.calls) == 6  # initial + 5 corrective retries


def test_agent_stream_llm_error_yields_error_and_done(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    client = FailingOpenAIClient()
    gen = Generator(settings, client=client)

    events = asyncio.run(
        _collect(
            run_agent_stream(
                question="What is hybrid retrieval?",
                generator=gen,
                registry=registry,
                memory=memory,
                settings=settings,
                trace=trace,
                max_iterations=3,
                conv_id="test_conv",
            )
        )
    )

    types = [e["type"] for e in events]
    assert types[0] == "step_start"
    assert "error" in types  # graceful event, not an exception
    assert "done" in types  # stream still terminates cleanly
    done = next(e for e in events if e["type"] == "done")
    assert "couldn't reach a conclusion" in done["answer"].lower()
    assert done["conversation_id"] == "test_conv"
    assert len(client.chat.completions.calls) == 6  # initial + 5 corrective retries


def test_system_prompt_blocks_native_tool_syntax(tmp_path):
    _settings, _index, _memory, registry = build_test_registry(tmp_path)
    prompt = build_system_prompt(registry, 6)
    assert "Native tool-call" in prompt
    assert '"arguments"' in prompt


# --- Grounding enforcement (deterministic loop rules) ----------------------


def test_grounding_enforces_search_on_direct_answer(tmp_path):
    """A first-turn final_answer without tools must be re-routed to search."""
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    always = json.dumps(
        {"thought": "I know this", "tool": None, "args": None, "final_answer": "From memory [1]."}
    )
    client = FakeOpenAIClient(always=always)
    gen = Generator(settings, client=client)

    result = run_loop(gen, settings, registry, memory, trace)

    assert result.success
    assert result.answer == "From memory [1]."
    assert result.tools_called == ["search_knowledge"]
    assert result.steps[0].tool_name == "search_knowledge"
    assert len(client.chat.completions.calls) == 2  # forced search, then accepted


def test_grounding_allows_greetings(tmp_path):
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    always = json.dumps(
        {"thought": "greet", "tool": None, "args": None, "final_answer": "Hello! How can I help?"}
    )
    client = FakeOpenAIClient(always=always)
    gen = Generator(settings, client=client)

    result = run_loop(gen, settings, registry, memory, trace, question="Hello!")

    assert result.success
    assert result.tools_called == []
    assert result.answer == "Hello! How can I help?"
    assert len(client.chat.completions.calls) == 1


def test_grounding_forces_run_python_for_code_requests(tmp_path):
    """Code requests: search first, then run the snippet, then answer."""
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    answer = (
        "Here is an MRR example:\n\n```python\n"
        "def mrr(ranks):\n    return sum(1 / r for r in ranks) / len(ranks)\n```"
    )
    always = json.dumps(
        {"thought": "code answer", "tool": None, "args": None, "final_answer": answer}
    )
    client = FakeOpenAIClient(always=always)
    gen = Generator(settings, client=client)

    result = run_loop(
        gen, settings, registry, memory, trace, question="Show me a code example of computing MRR"
    )

    assert result.success
    assert result.tools_called == ["search_knowledge", "run_python"]
    assert result.answer == answer


def test_grounding_searches_even_if_model_picked_run_python_first(tmp_path):
    """search_knowledge stays mandatory regardless of which tool came first."""
    settings, _index, memory, registry = build_test_registry(tmp_path)
    trace = Trace(settings.trace_dir)
    answer = "```python\nprint(1)\n```"
    script = [
        json.dumps(
            {"thought": "run", "tool": "run_python", "args": {"code": "print(1)"}, "final_answer": None}
        ),
        json.dumps({"thought": "done", "tool": None, "args": None, "final_answer": answer}),
        json.dumps({"thought": "done", "tool": None, "args": None, "final_answer": answer}),
    ]
    client = FakeOpenAIClient(script=script)
    gen = Generator(settings, client=client)

    result = run_loop(gen, settings, registry, memory, trace, question="Write a code example")

    assert result.success
    assert set(result.tools_called) == {"search_knowledge", "run_python"}


# --- Tolerant plan parsing + plan retries ----------------------------------


def test_extract_json_recovers_from_prose():
    raw = 'Sure! Here it is: {"thought": "t", "tool": "search_knowledge", "args": {"query": "x"}, "final_answer": null} done'
    obj = _extract_json(raw)
    assert obj is not None
    assert obj["tool"] == "search_knowledge"
    assert _extract_json("no json here") is None


def test_parse_decision_nudges_on_garbage():
    d = _parse_decision("complete garbage")
    assert d["tool"] is None and d["final_answer"] is None


class _FlakyCompletions:
    """Replays actions: Exception instances raise, str returns as content."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return FakeCompletion(action)


def _flaky_client(actions):
    from types import SimpleNamespace

    return SimpleNamespace(chat=SimpleNamespace(completions=_FlakyCompletions(actions)))


def test_plan_survives_json_rejection_and_empty_content(tmp_path):
    settings = make_env(tmp_path)
    reject = RuntimeError(
        "Error code: 400 - json_validate_failed with empty failed_generation"
    )
    good = json.dumps({"thought": "t", "tool": None, "args": None, "final_answer": "ok"})
    # attempt sequence: json, json, plain, json, plain, plain
    client = _flaky_client([reject, reject, "   ", good])
    gen = Generator(settings, client=client)

    resp, err = _plan(gen, settings, [{"role": "user", "content": "hi json"}])

    assert err is None
    assert resp is not None
    assert len(client.chat.completions.calls) == 4


def test_plan_fast_fails_on_daily_quota(tmp_path):
    settings = make_env(tmp_path)
    tpd = RuntimeError(
        "Error code: 429 - Rate limit reached for model on tokens per day (TPD): "
        "Limit 200000, Used 199305. Please try again in 8m31s."
    )
    client = _flaky_client([tpd])
    gen = Generator(settings, client=client)

    resp, err = _plan(gen, settings, [{"role": "user", "content": "hi json"}])

    assert resp is None
    assert err is not None and "TPD" in err
    assert len(client.chat.completions.calls) == 1  # no futile retries
