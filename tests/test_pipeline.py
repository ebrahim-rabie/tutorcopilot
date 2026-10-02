from app.index import Index
from app.pipeline import answer_question
from app.settings import Settings
from app.tracing import Trace
from tests.helpers import FakeGenerator, HashEmbedder


def _make_index():
    settings = Settings()
    settings.chunk_strategy = "recursive"
    embedder = HashEmbedder(dim=32)
    index = Index(settings, embedder)
    index.add_documents(
        {
            "retrieval.md": "Hybrid retrieval fuses BM25 and dense vectors using RRF with k=60.",
            "eval.md": "Golden datasets contain questions, reference answers, and expected sources.",
        }
    )
    return index


def test_answer_question_basic():
    index = _make_index()
    settings = Settings()
    gen = FakeGenerator(
        answer="Hybrid retrieval uses RRF to fuse BM25 and dense scores.", citations=[1]
    )
    trace = Trace("traces")
    result = answer_question("What is hybrid retrieval?", index, gen, settings, trace=trace)
    assert result.answer == "Hybrid retrieval uses RRF to fuse BM25 and dense scores."
    assert len(result.citations) == 1
    assert result.citations[0]["source"] == "retrieval.md"
    assert result.trace_id == trace.trace_id
    assert result.latency_ms > 0


def test_answer_question_no_rerank():
    index = _make_index()
    settings = Settings()
    settings.use_rerank = False
    gen = FakeGenerator(answer="RRF is the fusion method.", citations=[1])
    result = answer_question("What is RRF?", index, gen, settings, use_rerank=False)
    assert "RRF" in result.answer


def test_answer_question_top_k_override():
    index = _make_index()
    settings = Settings()
    gen = FakeGenerator(answer="test", citations=[])
    result = answer_question("question", index, gen, settings, top_k=1)
    assert len(result.retrieval_path) <= 1
