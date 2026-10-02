import numpy as np

from app.embedding import HashEmbedder
from app.index import Index
from app.settings import Settings


def _make_settings():
    s = Settings()
    s.chunk_strategy = "recursive"
    s.hybrid_top_k = 5
    s.final_top_k = 3
    s.rrf_k = 60
    return s


def test_hybrid_search_returns_results():
    settings = _make_settings()
    embedder = HashEmbedder(dim=32)
    index = Index(settings, embedder)
    index.add_documents(
        {
            "doc1.md": "This is about retrieval and embeddings in vector search.",
            "doc2.md": "BM25 uses term frequency and inverse document frequency.",
            "doc3.md": "Reranking with cross-encoders improves precision.",
        }
    )
    results = index.hybrid_search("vector search embeddings", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0
    for r in results:
        assert r.chunk.source in ("doc1.md", "doc2.md", "doc3.md")
        assert r.score > 0


def test_dense_search():
    settings = _make_settings()
    embedder = HashEmbedder(dim=32)
    index = Index(settings, embedder)
    index.add_documents({"a.md": "foo bar baz", "b.md": "hello world"})
    results = index.dense_search("foo bar", top_k=2)
    assert len(results) <= 2
    sources = [r.chunk.source for r in results]
    assert "a.md" in sources


def test_bm25_search():
    settings = _make_settings()
    embedder = HashEmbedder(dim=32)
    index = Index(settings, embedder)
    index.add_documents({"a.md": "foo bar baz", "b.md": "hello world"})
    results = index.bm25_search("hello", top_k=2)
    assert len(results) <= 2
    sources = [r.chunk.source for r in results]
    assert "b.md" in sources


def test_save_load_roundtrip(tmp_path):
    settings = _make_settings()
    embedder = HashEmbedder(dim=32)
    index = Index(settings, embedder)
    index.add_documents({"doc.md": "content for saving and loading test"})
    index.save(tmp_path)

    # New index, load
    index2 = Index(settings, embedder)
    ok = index2.load(tmp_path)
    assert ok
    assert len(index2.chunks) == len(index.chunks)
    assert np.allclose(index2._matrix, index._matrix)
    # Search works after load
    results = index2.hybrid_search("saving test", top_k=3)
    assert len(results) > 0
