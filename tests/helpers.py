import hashlib

import numpy as np

from app.embedding import BaseEmbedder


class HashEmbedder(BaseEmbedder):
    """Deterministic text-hash embedder for tests. No model downloads."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.model_name = "hash-embedder"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            vecs[i] = v / (np.linalg.norm(v) + 1e-9)
        return vecs


class FakeGenerator:
    def __init__(self, answer="test answer", citations=None):
        self.answer = answer
        self._citations = citations or []
        self.model = "fake-model"
        self.prompt_tokens = 10
        self.completion_tokens = 10
        self.latency_ms = 1.0
        self.confidence = 0.9
        self.parse_error = ""

    def available(self) -> bool:
        return True

    def generate(self, question, results):
        citations = []
        for c in self._citations:
            if 1 <= c <= len(results):
                r = results[c - 1]
                citations.append(
                    {
                        "index": c,
                        "source": r.chunk.source,
                        "section": r.chunk.section,
                        "chunk_id": r.chunk.id,
                        "text": r.chunk.text,
                        "score": r.score,
                    }
                )
        from types import SimpleNamespace

        return SimpleNamespace(
            answer=self.answer,
            citations=citations,
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_ms=self.latency_ms,
            confidence=self.confidence,
            parse_error=self.parse_error,
        )


class FakeChatMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChatChoice:
    def __init__(self, content: str):
        self.message = FakeChatMessage(content)


class FakeUsage:
    def __init__(self):
        self.prompt_tokens = 5
        self.completion_tokens = 7


class FakeCompletion:
    def __init__(self, content: str):
        self.choices = [FakeChatChoice(content)]
        self.usage = FakeUsage()


class FakeCompletions:
    """OpenAI-compatible chat.completions whose .create() replays scripts."""

    def __init__(self, script: list[str] | None = None, *, always: str | None = None):
        self.script = list(script or [])
        self.always = always
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs.get("messages", []))
        content = self.always if self.always is not None else self.script.pop(0)
        return FakeCompletion(content)


class FakeChat:
    def __init__(self, script=None, *, always=None):
        self.completions = FakeCompletions(script, always=always)


class FakeOpenAIClient:
    def __init__(self, script=None, *, always=None):
        self.chat = FakeChat(script, always=always)
