from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.loop import run_agent_loop
from app.generate import Generator, retry_transient
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.tools.registry import ToolRegistry
from app.tracing import Trace

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    conversation_id: str | None = None
    max_iterations: int = Field(default=6, ge=1, le=15)
    use_rerank: bool | None = None


class AgentResponse(BaseModel):
    answer: str
    conversation_id: str
    steps: list[dict]
    trace_id: str
    total_latency_ms: float
    tokens_used: dict
    tools_called: list[str]
    success: bool
    error: str | None = None


# Dependencies
def get_memory() -> MemoryManager:
    from app.main import app

    return app.state.memory


def get_tool_registry() -> ToolRegistry:
    from app.main import app

    return app.state.get_tool_registry_dep()


def get_generator() -> Generator:
    from app.main import app

    return app.state.get_generator_dep()


def get_settings() -> Settings:
    from app.main import app

    return app.state.settings


@router.post("/ask", response_model=AgentResponse)
async def agent_ask(
    req: AgentRequest,
    memory: MemoryManager = Depends(get_memory),
    registry: ToolRegistry = Depends(get_tool_registry),
    generator: Generator = Depends(get_generator),
    settings: Settings = Depends(get_settings),
):
    """Run the agent loop and return the final answer."""
    if not generator.available():
        raise HTTPException(503, "LLM not configured")

    conv_id = req.conversation_id or memory.start_conversation()
    trace = Trace(settings.trace_dir)
    try:
        result = await run_agent_loop(
            question=req.question,
            generator=generator,
            registry=registry,
            memory=memory,
            settings=settings,
            trace=trace,
            max_iterations=req.max_iterations,
            conv_id=conv_id,
        )
    finally:
        trace.save()

    return AgentResponse(
        answer=result.answer,
        conversation_id=conv_id,
        steps=[s.__dict__ for s in result.steps],
        trace_id=result.trace_id,
        total_latency_ms=result.total_latency_ms,
        tokens_used=result.tokens_used,
        tools_called=result.tools_called,
        success=result.success,
        error=result.error,
    )


@router.post("/stream")
async def agent_stream(
    req: AgentRequest,
    memory: MemoryManager = Depends(get_memory),
    registry: ToolRegistry = Depends(get_tool_registry),
    generator: Generator = Depends(get_generator),
    settings: Settings = Depends(get_settings),
):
    """Stream agent execution as Server-Sent Events."""
    if not generator.available():
        raise HTTPException(503, "LLM not configured")

    conv_id = req.conversation_id or memory.start_conversation()
    trace = Trace(settings.trace_dir)

    async def event_generator() -> AsyncGenerator[str, None]:
        from app.streaming.sse import sse_generator

        async for chunk in sse_generator(
            req.question,
            generator,
            registry,
            memory,
            settings,
            trace,
            max_iterations=req.max_iterations,
            conv_id=conv_id,
        ):
            yield chunk
        trace.save()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    memory: MemoryManager = Depends(get_memory),
):
    """Get conversation history."""
    # Load from episodic memory
    conv = memory.episodic.load_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.delete("/conversations/{conversation_id}")
async def end_conversation(
    conversation_id: str,
    memory: MemoryManager = Depends(get_memory),
):
    """End and persist a conversation."""
    if memory.current_conv_id == conversation_id:
        memory.end_conversation()
    return {"status": "ended", "conversation_id": conversation_id}


@router.get("/memory/weak-topics")
async def get_weak_topics(memory: MemoryManager = Depends(get_memory)):
    """Get topics the user struggles with."""
    return {"weakest_topic": memory.get_weakest_topic()}


# --- Voice: speech-to-text -----------------------------------------------

_AUDIO_MIME_EXT = {
    "audio/webm": "webm",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
}


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    generator: Generator = Depends(get_generator),
    settings: Settings = Depends(get_settings),
):
    """Speech -> text via the configured STT model (Groq Whisper).

    Metered separately from LLM tokens (audio seconds/day), so it stays
    usable while the chat-completion quota is spent.
    """
    if not generator.available():
        raise HTTPException(503, "LLM not configured")
    mime = (audio.content_type or "").lower()
    if not mime.startswith("audio/"):
        raise HTTPException(415, f"expected audio/*, got {mime or 'unknown'}")
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    max_bytes = settings.stt_max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"audio exceeds {settings.stt_max_upload_mb} MB")
    ext = _AUDIO_MIME_EXT.get(mime, "webm")

    try:
        result = retry_transient(
            lambda: generator.client.audio.transcriptions.create(
                model=settings.stt_model,
                file=(f"audio.{ext}", data, mime),
                response_format="json",
            )
        )
    except Exception as exc:  # API/transport failure must degrade, not crash
        detail = str(exc)
        status = 503 if "429" in detail or "rate limit" in detail.lower() else 502
        raise HTTPException(status_code=status, detail=f"transcription failed: {detail[:300]}") from exc
    return {"text": (result.text or "").strip(), "model": settings.stt_model}


# --- Voice: text-to-speech ------------------------------------------------


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    voice: str | None = None


async def _edge_synth(text: str, voice: str) -> bytes:
    """Render text to mp3 bytes with edge-tts (lazy import, no key needed)."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    chunks = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            chunks.append(chunk["data"])
    if not chunks:
        raise RuntimeError("edge-tts returned no audio")
    return b"".join(chunks)


@router.post("/speak")
async def speak(req: SpeakRequest, settings: Settings = Depends(get_settings)):
    """Text -> speech. Returns audio/mpeg for direct <audio> playback."""
    if settings.tts_provider != "edge":
        raise HTTPException(503, "server TTS not configured (TC_TTS_PROVIDER=edge)")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    try:
        audio = await _edge_synth(text, req.voice or settings.tts_voice)
    except Exception as exc:  # provider outage must degrade, not crash
        raise HTTPException(502, f"TTS failed: {str(exc)[:300]}") from exc
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )
