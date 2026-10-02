from __future__ import annotations

import hashlib

import numpy as np


class BaseEmbedder:
    dim: int = 0

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers embedder, runs fully offline after first download."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        encoded = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(encoded, dtype=np.float32)


class ApiEmbedder(BaseEmbedder):
    """OpenAI-compatible embeddings endpoint (useful for dense-only cloud options)."""

    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
        self.dim = 1536

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class HashEmbedder(BaseEmbedder):
    """Deterministic text-hash vectors. Use in tests when no model download is allowed."""

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


def get_embedder(settings) -> BaseEmbedder:
    mode = (settings.embedding_mode or "local").lower()
    if mode == "api":
        import openai

        client = openai.OpenAI(
            api_key=settings.llm_api_key or "unset",
            base_url=settings.llm_base_url or None,
        )
        return ApiEmbedder(client, model=settings.embedding_api_model)
    return LocalEmbedder(model_name=settings.embedding_model)
