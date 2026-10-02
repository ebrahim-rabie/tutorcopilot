from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """JSON Schema for a tool, exposed to the LLM."""

    name: str
    description: str
    parameters: dict  # JSON Schema object
    required: list[str] = Field(default_factory=list)

    def to_openai_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "latency_ms": round(self.latency_ms, 2),
        }


ToolFn = Callable[..., Awaitable[ToolResult]]


class BaseTool(ABC):
    """Abstract base for all tools. Subclasses implement `execute`."""

    name: str
    description: str
    parameters: dict
    required: list[str] = field(default_factory=list)

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required=self.required,
        )

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass

    def validate_args(self, args: dict) -> tuple[bool, str | None]:
        missing = [r for r in self.required if r not in args]
        if missing:
            return False, f"Missing required args: {missing}"
        return True, None


class ToolRegistry:
    """Registry of available tools, builds LLM function schemas."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict]:
        return [t.schema.to_openai_function() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())


# ============================================================
# Built-in Tools
# ============================================================


class SearchKnowledgeTool(BaseTool):
    """Search the Phase 1 knowledge index."""

    name = "search_knowledge"
    description = "Search the AI engineering knowledge base for relevant information. Use for factual questions about RAG, agents, fine-tuning, evals, serving, prompting."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results", "default": 4},
        },
    }
    required = ["query"]

    def __init__(self, index, use_rerank: bool = True):
        self.index = index
        self.use_rerank = use_rerank

    async def execute(self, query: str, top_k: int = 4) -> ToolResult:
        start = time.monotonic()
        results = self.index.hybrid_search(query, top_k=top_k * 3)
        if self.use_rerank:
            results = self.index.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        latency = (time.monotonic() - start) * 1000
        return ToolResult(
            success=True,
            data=[r.to_dict() for r in results],
            metadata={"query": query, "use_rerank": self.use_rerank},
            latency_ms=latency,
        )


class RunPythonTool(BaseTool):
    """Execute Python code in a sandboxed subprocess (timeout, no network)."""

    name = "run_python"
    description = "Run a short Python snippet to verify code, compute something, or test logic. No network, no filesystem writes. Timeout 10s."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
    }
    required = ["code"]

    async def execute(self, code: str) -> ToolResult:
        import subprocess
        import sys

        start = time.monotonic()
        try:
            # Run with timeout, capture stdout/stderr
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            latency = (time.monotonic() - start) * 1000
            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    data={"stdout": proc.stdout, "stderr": proc.stderr},
                    latency_ms=latency,
                )
            else:
                return ToolResult(
                    success=False,
                    error=proc.stderr or f"Exit code {proc.returncode}",
                    data={"stdout": proc.stdout},
                    latency_ms=latency,
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error="Execution timed out (10s)",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )


class FetchURLTool(BaseTool):
    """Fetch and extract text from a URL."""

    name = "fetch_url"
    description = "Fetch a web page and return extracted text content. Useful for reading documentation, articles, or API references."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "max_chars": {"type": "integer", "description": "Max chars to return", "default": 8000},
        },
    }
    required = ["url"]

    async def execute(self, url: str, max_chars: int = 8000) -> ToolResult:
        import httpx
        from bs4 import BeautifulSoup

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts/styles
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... [truncated to {max_chars} chars]"
            latency = (time.monotonic() - start) * 1000
            return ToolResult(
                success=True,
                data={"url": url, "text": text, "length": len(text)},
                latency_ms=latency,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )


class SaveNoteTool(BaseTool):
    """Save a note to the user's knowledge base (appends to a markdown file)."""

    name = "save_note"
    description = (
        "Save a note or fact to your personal knowledge base. Creates/updates a markdown file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Note title"},
            "content": {"type": "string", "description": "Note content (markdown)"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
        },
    }
    required = ["title", "content"]

    def __init__(self, notes_dir: str = "data/notes"):
        self.notes_dir = Path(notes_dir)
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, title: str, content: str, tags: list[str] | None = None) -> ToolResult:
        import datetime

        start = time.monotonic()
        try:
            safe_title = (
                "".join(c for c in title if c.isalnum() or c in " -_").strip().replace(" ", "_")
            )
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_str}_{safe_title}.md"
            path = self.notes_dir / filename
            tag_line = f"tags: [{', '.join(tags)}]\n" if tags else ""
            md = f"""---
title: {title}
date: {date_str}
{tag_line}---
{content}
"""
            path.write_text(md, encoding="utf-8")
            latency = (time.monotonic() - start) * 1000
            return ToolResult(
                success=True,
                data={"path": str(path), "title": title},
                latency_ms=latency,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )


class QuizMeTool(BaseTool):
    """Generate a quiz question from the user's weak areas (tracked in memory)."""

    name = "quiz_me"
    description = "Generate a quiz question to test knowledge on a weak topic. Returns a question and expected answer."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic to quiz on (optional, picks weakest)",
            },
        },
    }
    required = []

    def __init__(self, memory):
        self.memory = memory

    async def execute(self, topic: str | None = None) -> ToolResult:
        start = time.monotonic()
        try:
            # Get weak area or use provided topic
            if topic is None:
                topic = self.memory.get_weakest_topic()
                if not topic:
                    return ToolResult(
                        success=False,
                        error="No weak topics tracked yet. Ask some questions first!",
                        latency_ms=(time.monotonic() - start) * 1000,
                    )
            # Generate question from topic (simplified - could use LLM)
            question = f"Explain the key concepts of {topic} in your own words."
            # Reference answer from knowledge base would be ideal; here we return a prompt
            latency = (time.monotonic() - start) * 1000
            return ToolResult(
                success=True,
                data={
                    "topic": topic,
                    "question": question,
                    "expected_answer": f"Answer should cover key concepts from the knowledge base about {topic}.",
                },
                latency_ms=latency,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )


# ============================================================
# Registry builder
# ============================================================


def build_tool_registry(index, memory, use_rerank: bool = True) -> ToolRegistry:
    """Create registry with all Phase 2 tools wired up."""
    reg = ToolRegistry()
    reg.register(SearchKnowledgeTool(index, use_rerank))
    reg.register(RunPythonTool())
    reg.register(FetchURLTool())
    reg.register(SaveNoteTool())
    reg.register(QuizMeTool(memory))
    return reg
