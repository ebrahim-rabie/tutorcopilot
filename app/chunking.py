from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .embedding import BaseEmbedder

SPLITTERS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str = ""
    char_count: int = 0
    token_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.token_count = len(self.text.split())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split text respecting separators; falls back to smaller chunks, then hard cuts."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [
            text[i : i + chunk_size].strip()
            for i in range(0, len(text), chunk_size)
            if text[i : i + chunk_size].strip()
        ]
    sep = separators[0]
    parts = text.split(sep)
    if len(parts) <= 1:
        return _recursive_split(text, chunk_size, separators[1:])

    chunks: list[str] = []
    buffer = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if buffer and len(buffer) + len(sep) + len(piece) <= chunk_size:
            buffer = buffer + sep + piece
        else:
            if buffer:
                chunks.append(buffer)
            buffer = ""
            if len(piece) > chunk_size:
                chunks.extend(_recursive_split(piece, chunk_size, separators[1:]))
            else:
                buffer = piece
    if buffer:
        chunks.append(buffer)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out: list[str] = []
    prev = ""
    for chunk in chunks:
        if not out:
            out.append(chunk)
        else:
            out.append((prev[-overlap:] + " " + chunk).strip() if prev else chunk)
        prev = chunk
    return out


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str, section: str = "") -> list[Chunk]:
        parts = _recursive_split(text, self.chunk_size, SPLITTERS)
        parts = _apply_overlap(parts, self.overlap)
        return [
            Chunk(id=f"{source}#{i:04d}", text=p, source=source, section=section)
            for i, p in enumerate(parts)
            if p.strip()
        ]


class MarkdownChunker:
    """Split by markdown headings first, then recursively chunk inside each section."""

    name = "markdown"

    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str, section: str = "") -> list[Chunk]:
        sections: list[tuple[str, str]] = []
        current_title = section
        body: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                if body or current_title:
                    sections.append((current_title, "\n".join(body)))
                    body = []
                current_title = m.group(2).strip()
            else:
                body.append(line)
        sections.append((current_title, "\n".join(body)))

        chunks: list[Chunk] = []
        for title, body_text in sections:
            if not body_text.strip():
                continue
            parts = _recursive_split(body_text, self.chunk_size, SPLITTERS)
            parts = _apply_overlap(parts, self.overlap)
            for i, p in enumerate(parts):
                if p.strip():
                    chunks.append(
                        Chunk(id=f"{source}#{i:04d}", text=p.strip(), source=source, section=title)
                    )
        return chunks


class SemanticChunker:
    """Break sentences, merge while consecutive cosine similarity stays above threshold."""

    name = "semantic"

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float = 0.7,
        min_chars: int = 80,
        max_chars: int = 800,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.min_chars = min_chars
        self.max_chars = max_chars

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]

    def chunk_text(self, text: str, source: str, section: str = "") -> list[Chunk]:
        import numpy as np

        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [Chunk(id=f"{source}#0000", text=sentences[0], source=source, section=section)]

        embs = self.embedder.embed_texts(sentences)
        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        current_vec = embs[0].copy()
        for i in range(1, len(sentences)):
            sim = float(np.dot(current_vec, embs[i]))
            joins = " ".join(current)
            if sim >= self.threshold and len(joins) + len(sentences[i]) <= self.max_chars:
                current.append(sentences[i])
                n = len(current)
                current_vec = (current_vec * (n - 1) + embs[i]) / n
                current_vec = current_vec / (np.linalg.norm(current_vec) + 1e-9)
            else:
                groups.append(current)
                current = [sentences[i]]
                current_vec = embs[i].copy()
        groups.append(current)

        # Merge tiny groups into neighbouring big chunks.
        merged: list[str] = []
        buffer = ""
        for group in groups:
            candidate = " ".join(group)
            if buffer and len(buffer) + len(candidate) <= self.max_chars:
                buffer += " " + candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = candidate
        if buffer:
            merged.append(buffer)

        return [
            Chunk(id=f"{source}#{i:04d}", text=c.strip(), source=source, section=section)
            for i, c in enumerate(merged)
            if c.strip() and len(c.strip()) >= self.min_chars
        ]


def get_chunker(settings, embedder: BaseEmbedder | None = None):
    strategy = (settings.chunk_strategy or "markdown").lower()
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        return SemanticChunker(embedder=embedder)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    return MarkdownChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
