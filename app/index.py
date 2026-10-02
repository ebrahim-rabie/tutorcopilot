from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.linalg as LA
from rank_bm25 import BM25Okapi

from .chunking import Chunk, get_chunker
from .embedding import BaseEmbedder
from .rerank import CrossEncoderReranker
from .settings import Settings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "source": self.chunk.source,
            "section": self.chunk.section,
            "chunk_id": self.chunk.id,
            "text": self.chunk.text,
            "score": float(self.score),
            "dense_score": float(self.dense_score),
            "bm25_score": float(self.bm25_score),
            "rank": self.rank,
        }


class Index:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    @property
    def ready(self) -> bool:
        return len(self.chunks) > 0 and self._matrix is not None

    # ---------- build ----------
    def add_documents(self, doc_texts: dict[str, str]) -> None:
        chunker = get_chunker(self.settings, embedder=self.embedder)
        for source, text in doc_texts.items():
            for chunk in chunker.chunk_text(text, source=source):
                self.chunks.append(chunk)
        self._build_embeddings()
        self._build_bm25()

    def build_from_directory(self, sources_dir: Path | None = None) -> None:
        sources_dir = Path(sources_dir or self.settings.sources_dir)
        texts: dict[str, str] = {}
        for pattern in ("*.md", "*.txt"):
            for path in sorted(sources_dir.glob(pattern)):
                texts[path.name] = path.read_text(encoding="utf-8")
        if not texts:
            raise FileNotFoundError(f"No source documents found in {sources_dir}")
        self.add_documents(texts)

    def _build_embeddings(self) -> None:
        if not self.chunks:
            self._matrix = np.zeros((0, self.embedder.dim or 384), dtype=np.float32)
            return
        self._matrix = self.embedder.embed_texts([c.text for c in self.chunks])

    def _build_bm25(self) -> None:
        self._corpus_tokens = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    # ---------- search ----------
    @staticmethod
    def _cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
        q = query_vec / (LA.norm(query_vec) + 1e-9)
        norms = LA.norm(matrix, axis=1) + 1e-9
        return (matrix / norms[:, None]) @ q

    def dense_search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        top_k = top_k or self.settings.hybrid_top_k
        sims = self._cosine_scores(self._matrix, self.embedder.embed(query))
        order = np.argsort(-sims)
        return [
            SearchResult(
                chunk=self.chunks[i], score=float(sims[i]), dense_score=float(sims[i]), rank=r
            )
            for r, i in enumerate(order[:top_k])
        ]

    def bm25_search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        top_k = top_k or self.settings.hybrid_top_k
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)
        return [
            SearchResult(
                chunk=self.chunks[i], score=float(scores[i]), bm25_score=float(scores[i]), rank=r
            )
            for r, i in enumerate(order[:top_k])
        ]

    def hybrid_search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """BM25 + dense embeddings fused with Reciprocal Rank Fusion."""
        top_k = top_k or self.settings.hybrid_top_k
        sims = self._cosine_scores(self._matrix, self.embedder.embed(query))
        dense_order = np.argsort(-sims)
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(-bm25_scores)

        candidate_count = max(top_k * 3, 30)
        k = self.settings.rrf_k
        rrf: dict[int, float] = {}
        for pos, i in enumerate(dense_order[:candidate_count]):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (k + pos + 1)
        for pos, i in enumerate(bm25_order[:candidate_count]):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (k + pos + 1)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=score,
                dense_score=float(sims[i]),
                bm25_score=float(bm25_scores[i]),
                rank=rank,
            )
            for rank, (i, score) in enumerate(fused)
        ]

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        top_k = top_k or self.settings.final_top_k
        if not results:
            return results
        if self.reranker is None:
            return results[:top_k]
        pairs = [(query, r.chunk.text) for r in results]
        scores = self.reranker.predict(pairs)
        combined = sorted(zip(results, scores, strict=False), key=lambda x: x[1], reverse=True)
        out: list[SearchResult] = []
        for rank, (result, score) in enumerate(combined[:top_k]):
            result.score = score
            result.rank = rank
            out.append(result)
        return out

    # ---------- persistence ----------
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.settings.index_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(path / "matrix.npy", self._matrix)
        (path / "corpus_tokens.json").write_text(json.dumps(self._corpus_tokens), encoding="utf-8")
        meta = {
            "embedding_model": getattr(self.embedder, "model_name", None)
            or getattr(self.embedder, "model", None),
            "chunk_strategy": self.settings.chunk_strategy,
            "chunk_size": self.settings.chunk_size,
            "n_chunks": len(self.chunks),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def load(self, path: str | Path | None = None) -> bool:
        path = Path(path or self.settings.index_dir)
        chunks_file = path / "chunks.json"
        matrix_file = path / "matrix.npy"
        tokens_file = path / "corpus_tokens.json"
        if not all(p.exists() for p in (chunks_file, matrix_file, tokens_file)):
            return False
        self.chunks = [
            Chunk.from_dict(d) for d in json.loads(chunks_file.read_text(encoding="utf-8"))
        ]
        self._matrix = np.load(matrix_file).astype(np.float32) if self.chunks else np.zeros((0, 1))
        self._corpus_tokens = json.loads(tokens_file.read_text(encoding="utf-8"))
        self._bm25 = BM25Okapi(self._corpus_tokens)
        return True


def load_or_build_index(
    settings: Settings,
    embedder: BaseEmbedder,
    reranker: CrossEncoderReranker | None = None,
) -> Index:
    index = Index(settings, embedder, reranker)
    if not index.load():
        index.build_from_directory()
        index.save()
    return index
