import pytest
from fastapi.testclient import TestClient

from app.index import Index
from app.main import create_app
from app.settings import Settings
from tests.helpers import FakeGenerator, HashEmbedder


@pytest.fixture
def app():
    settings = Settings()
    settings.chunk_strategy = "recursive"
    return create_app(settings)


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health(app, client):
    # Override deps with fake index/generator
    settings = Settings()
    settings.chunk_strategy = "recursive"
    embedder = HashEmbedder(dim=32)
    fake_index = Index(settings, embedder)
    fake_index.add_documents({"a.md": "test content"})
    fake_gen = FakeGenerator(answer="ok")

    # get the dependency functions from app state
    get_index = app.state.get_index_dep
    get_gen = app.state.get_generator_dep

    app.dependency_overrides[get_index] = lambda: fake_index
    app.dependency_overrides[get_gen] = lambda: fake_gen

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index_chunks"] == 1
    assert data["llm_configured"] is True


def test_ask_endpoint(app, client):
    settings = Settings()
    settings.chunk_strategy = "recursive"
    embedder = HashEmbedder(dim=32)
    fake_index = Index(settings, embedder)
    fake_index.add_documents({"doc.md": "Hybrid retrieval fuses BM25 and dense via RRF."})
    fake_gen = FakeGenerator(answer="It uses RRF with k=60.", citations=[1])

    get_index = app.state.get_index_dep
    get_gen = app.state.get_generator_dep

    app.dependency_overrides[get_index] = lambda: fake_index
    app.dependency_overrides[get_gen] = lambda: fake_gen

    resp = client.post("/ask", json={"question": "How does hybrid retrieval work?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "It uses RRF with k=60."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source"] == "doc.md"
    assert data["trace_id"]
    assert data["retrieval_path"]


def test_ask_missing_llm(app, client):
    settings = Settings()
    settings.llm_api_key = ""  # hermetic: never trust local .env
    embedder = HashEmbedder(dim=32)
    fake_index = Index(settings, embedder)
    fake_index.add_documents({"a.md": "content"})
    # Generator without client -> unavailable
    from app.generate import Generator

    fake_gen = Generator(settings)  # no client -> unavailable

    get_index = app.state.get_index_dep
    get_gen = app.state.get_generator_dep

    app.dependency_overrides[get_index] = lambda: fake_index
    app.dependency_overrides[get_gen] = lambda: fake_gen

    resp = client.post("/ask", json={"question": "test"})
    assert resp.status_code == 503
    assert "LLM not configured" in resp.json()["detail"]


def test_trace_not_found(app, client):
    resp = client.get("/traces/nonexistent123")
    assert resp.status_code == 404
