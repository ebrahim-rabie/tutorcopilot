from __future__ import annotations

from dataclasses import dataclass

from .generate import Generator, retry_transient
from .index import Index
from .settings import Settings
from .tracing import Trace


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[dict]
    retrieval_path: list[dict]
    top_chunks: list[dict]
    trace_id: str
    latency_ms: float
    usage: dict
    model: str
    confidence: float = 0.0


def answer_question(
    question: str,
    index: Index,
    generator: Generator,
    settings: Settings,
    top_k: int | None = None,
    use_rerank: bool | None = None,
    trace: Trace | None = None,
) -> AnswerResult:
    top_k = top_k or settings.final_top_k
    use_rerank = settings.use_rerank if use_rerank is None else use_rerank

    retrieved = index.hybrid_search(question)
    if trace:
        trace.log("hybrid_search", {"query": question, "results": [r.to_dict() for r in retrieved]})

    if use_rerank:
        retrieved = index.rerank(question, retrieved, top_k=top_k)
        if trace:
            trace.log("rerank", {"query": question, "results": [r.to_dict() for r in retrieved]})
    else:
        retrieved = retrieved[:top_k]

    if trace:
        trace.log("retrieval_done", {"n_results": len(retrieved)})

    gen = retry_transient(lambda: generator.generate(question, retrieved))
    if trace:
        trace.log(
            "generate",
            {
                "model": gen.model,
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "latency_ms": gen.latency_ms,
                "answer": gen.answer,
                "citations": gen.citations,
                "confidence": gen.confidence,
            },
        )

    return AnswerResult(
        question=question,
        answer=gen.answer,
        citations=gen.citations,
        retrieval_path=[r.to_dict() for r in retrieved],
        top_chunks=[r.to_dict() for r in retrieved[:3]],
        trace_id=trace.trace_id if trace else "",
        latency_ms=gen.latency_ms,
        usage={"prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens},
        model=gen.model,
        confidence=gen.confidence,
    )
