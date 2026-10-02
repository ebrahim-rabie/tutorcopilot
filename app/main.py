from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .embedding import get_embedder
from .generate import Generator
from .index import Index, load_or_build_index
from .pipeline import answer_question
from .rerank import CrossEncoderReranker
from .settings import Settings
from .tracing import Trace, load_trace

# Phase 2 imports
try:
    from .agent.routes import router as agent_router
    from .memory.manager import MemoryManager
    from .tools.registry import ToolRegistry, build_tool_registry

    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="User question")
    top_k: int | None = Field(default=None, ge=1, le=10, description="Override final context size")
    use_rerank: bool | None = Field(default=None, description="Override rerank toggle")


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    confidence: float
    latency_ms: float
    usage: dict
    model: str
    trace_id: str
    retrieval_path: list[dict]
    top_chunks: list[dict]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="TutorCopilot",
        version="0.2.0",
        description="Eval-driven RAG + Agent for AI engineers.",
    )
    app.state.settings = settings

    # Core dependencies (Phase 1)
    def get_index_dep() -> Index:
        if not hasattr(app.state, "index"):
            embedder = get_embedder(settings)
            reranker = (
                CrossEncoderReranker(settings.reranker_model) if settings.use_rerank else None
            )
            app.state.index = load_or_build_index(settings, embedder, reranker)
        return app.state.index

    def get_generator_dep() -> Generator:
        if not hasattr(app.state, "generator"):
            app.state.generator = Generator(settings)
        return app.state.generator

    app.state.get_index_dep = get_index_dep
    app.state.get_generator_dep = get_generator_dep

    # Phase 2 dependencies (lazy)
    if PHASE2_AVAILABLE:
        app.state.memory = MemoryManager()

        def get_tool_registry_dep() -> ToolRegistry:
            if not hasattr(app.state, "tool_registry"):
                idx = get_index_dep()
                app.state.tool_registry = build_tool_registry(
                    index=idx,
                    memory=app.state.memory,
                    use_rerank=settings.use_rerank,
                )
            return app.state.tool_registry

        app.state.get_tool_registry_dep = get_tool_registry_dep

    # Health endpoint
    @app.get("/health")
    def health(
        index: Index = Depends(get_index_dep), generator: Generator = Depends(get_generator_dep)
    ):
        phase2 = "available" if PHASE2_AVAILABLE else "unavailable"
        return {
            "status": "ok",
            "phase": 2 if PHASE2_AVAILABLE else 1,
            "phase2": phase2,
            "index_chunks": len(index.chunks),
            "llm_configured": generator.available(),
            "stt_configured": generator.available(),
            "tts_provider": settings.tts_provider,
        }

    # Phase 1: /ask (RAG only)
    @app.post("/ask")
    def ask(
        req: AskRequest,
        index: Index = Depends(get_index_dep),
        generator: Generator = Depends(get_generator_dep),
    ):
        trace = Trace(settings.trace_dir)
        try:
            result = answer_question(
                req.question,
                index,
                generator,
                settings,
                top_k=req.top_k,
                use_rerank=req.use_rerank,
                trace=trace,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            trace.save()
        return AskResponse(
            question=result.question,
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            usage=result.usage,
            model=result.model,
            trace_id=result.trace_id,
            retrieval_path=result.retrieval_path,
            top_chunks=result.top_chunks,
        )

    @app.get("/traces/{trace_id}")
    def get_trace(trace_id: str):
        trace = load_trace(settings.trace_dir, trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return trace

    # Phase 2: Agent endpoints
    if PHASE2_AVAILABLE:
        app.include_router(agent_router)

    # Frontend: static assets + SPA entry (registered last, API routes win)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
