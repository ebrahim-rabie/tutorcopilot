from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal

from app.agent.loop import run_agent_stream
from app.generate import Generator
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.tools.registry import ToolRegistry
from app.tracing import Trace

EventType = Literal[
    "step_start",
    "thought",
    "tool_call",
    "tool_result",
    "nudge",
    "final_answer",
    "max_iterations",
    "done",
    "error",
]


@dataclass
class StreamEvent:
    type: EventType
    data: dict
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=lambda: __import__("time").time())


async def stream_agent(
    question: str,
    generator: Generator,
    registry: ToolRegistry,
    memory: MemoryManager,
    settings: Settings,
    trace: Trace,
    max_iterations: int = 6,
    conv_id: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Convert run_agent_stream to StreamEvent objects."""
    async for event in run_agent_stream(
        question, generator, registry, memory, settings, trace, max_iterations, conv_id
    ):
        yield StreamEvent(type=event["type"], data=event)


def format_sse(event: StreamEvent) -> str:
    """Format as Server-Sent Events."""
    return f"event: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


async def sse_generator(
    question: str,
    generator: Generator,
    registry: ToolRegistry,
    memory: MemoryManager,
    settings: Settings,
    trace: Trace,
    max_iterations: int = 6,
    conv_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted strings for FastAPI StreamingResponse."""
    async for event in stream_agent(
        question, generator, registry, memory, settings, trace, max_iterations, conv_id
    ):
        yield format_sse(event)
    # End marker
    yield "event: close\ndata: {}\n\n"
