from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .index import SearchResult
from .settings import Settings

SYSTEM_PROMPT = """You are TutorCopilot, a grounded assistant for AI engineers.

Rules:
- Answer the user's question using ONLY the numbered context blocks below.
- Support claims by citing the context with inline tags like [1] or [2, 3].
- If the context does not contain the answer, say so explicitly and do not invent facts.
- Keep the answer focused and under 250 words.

Return a single JSON object with exactly these keys:
- "answer": your answer text, with [n] citations inline
- "citations": the context numbers you actually used, e.g. ["1","3"]
- "confidence": a number from 0.0 to 1.0 estimating how well the context supports your answer"""


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict]
    raw: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    confidence: float = 0.0
    parse_error: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "raw": self.raw,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "confidence": self.confidence,
            "parse_error": self.parse_error,
        }


def _extract_json(text: str) -> tuple[dict | None, str]:
    """Parse model output, tolerating markdown fences and stray text."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None, "output is not a JSON object"
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else None, "output is not a JSON object"
            except json.JSONDecodeError:
                return None, text[:200]
        return None, text[:200]


def retry_transient(fn, attempts: int = 3, base_delay: float = 1.5):
    """Run an LLM call, retrying transient upstream failures.

    Proxies and free-tier APIs occasionally reset connections mid-request;
    evals and the RAG pipeline must succeed or degrade, never crash on a blip.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts - 1:
                time.sleep(base_delay * (attempt + 1))
    raise last_exc


def _build_client(settings: Settings):
    if not settings.llm_api_key:
        return None
    import openai

    return openai.OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
    )


class Generator:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client if client is not None else _build_client(settings)

    def available(self) -> bool:
        return self.client is not None

    def generate(self, question: str, results: list[SearchResult]) -> GenerationResult:
        if self.client is None:
            raise RuntimeError(
                "LLM not configured: set TC_LLM_API_KEY (and TC_LLM_BASE_URL / TC_LLM_MODEL) in .env"
            )
        context = "\n\n".join(f"[{i + 1}] {r.chunk.text}" for i, r in enumerate(results))
        user = f"Question:\n{question}\n\nContext:\n{context}"

        start = time.monotonic()
        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.monotonic() - start) * 1000
        raw = (resp.choices[0].message.content or "").strip()
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0

        parsed, parse_error = _extract_json(raw)
        answer = raw
        confidence = 0.0
        citations: list[dict] = []
        if parsed:
            answer = str(parsed.get("answer") or raw)
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            for c in parsed.get("citations", []):
                try:
                    idx = int(c)
                except (TypeError, ValueError):
                    continue
                if 1 <= idx <= len(results):
                    r = results[idx - 1]
                    citations.append(
                        {
                            "index": idx,
                            "source": r.chunk.source,
                            "section": r.chunk.section,
                            "chunk_id": r.chunk.id,
                            "text": r.chunk.text,
                            "score": float(r.score),
                        }
                    )

        return GenerationResult(
            answer=answer,
            citations=citations,
            raw=raw,
            model=self.settings.llm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            confidence=confidence,
            parse_error=parse_error,
        )


JUDGE_SYSTEM = """You are a strict evaluation judge. Rate the assistant's answer on two aspects, 1 (terrible) to 5 (excellent):

1. faithfulness: is EVERY claim in the answer supported by the provided context? Hallucinated or unsupported claims must score low regardless of how plausible they sound. If the answer correctly says the context lacks the information, that is not a faithfulness failure.
2. relevance: does the answer directly and completely address the question?

Return a JSON object with exactly:
{"faithfulness": int, "faithfulness_reason": str, "relevance": int, "relevance_reason": str}"""


class Judge:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client if client is not None else _build_client(settings)
        self.model = settings.llm_judge_model or settings.llm_model

    def available(self) -> bool:
        return self.client is not None

    def rate_grounded(self, question: str, answer: str, context_text: str) -> dict:
        if self.client is None:
            return {}
        user = (
            f"Context:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer under review:\n{answer}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        parsed, _err = _extract_json(resp.choices[0].message.content)
        if not parsed:
            return {}
        try:
            ff = int(parsed.get("faithfulness", 0))
        except (TypeError, ValueError):
            ff = 0
        try:
            rel = int(parsed.get("relevance", 0))
        except (TypeError, ValueError):
            rel = 0
        return {
            "faithfulness": ff,
            "faithfulness_reason": str(parsed.get("faithfulness_reason", "")),
            "relevance": rel,
            "relevance_reason": str(parsed.get("relevance_reason", "")),
        }
