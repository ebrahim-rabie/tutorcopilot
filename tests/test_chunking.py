from app.chunking import Chunk, MarkdownChunker, RecursiveChunker, _apply_overlap, _recursive_split


def test_recursive_split_basic():
    text = "This is a short text."
    out = _recursive_split(text, 500, ["\n\n", "\n", ". ", " "])
    assert len(out) == 1
    assert out[0] == text


def test_recursive_split_long():
    text = "Sentence one. " * 50  # ~800 chars
    out = _recursive_split(text, 300, ["\n\n", "\n", ". ", " "])
    assert len(out) > 1
    assert all(len(c) <= 300 for c in out)
    # Reconstructed text should be close to original (whitespace handling may differ)
    reconstructed = " ".join(out).strip()
    original = " ".join(text.split()).strip()
    assert reconstructed == original or len(reconstructed) >= len(original) * 0.9


def test_overlap_application():
    chunks = ["chunk one", "chunk two", "chunk three"]
    out = _apply_overlap(chunks, overlap=4)
    assert len(out) == 3
    assert out[0] == "chunk one"
    assert out[1].startswith("one")  # overlap from prev
    assert out[2].startswith("two")


def test_recursive_chunker_basic():
    chunker = RecursiveChunker(chunk_size=100, overlap=10)
    text = "First paragraph.\n\nSecond paragraph here.\n\nThird and final. " * 5
    chunks = chunker.chunk_text(text, "test.md", "intro")
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source == "test.md"
        assert c.section == "intro"
        assert c.char_count > 0


def test_markdown_chunker_headings():
    chunker = MarkdownChunker(chunk_size=200, overlap=20)
    md = """# Introduction
This is the intro section with some content.

## Getting Started
Steps to begin.

### Prerequisites
You need Python.
"""
    chunks = chunker.chunk_text(md, "guide.md")
    sections = [c.section for c in chunks]
    assert "Introduction" in sections
    assert "Getting Started" in sections
    assert "Prerequisites" in sections
    # Each chunk carries its section title
    for c in chunks:
        assert c.section in ["Introduction", "Getting Started", "Prerequisites"]


def test_chunk_dataclass():
    ch = Chunk(id="a#1", text="hello", source="a.md", section="sec")
    d = ch.to_dict()
    assert d["id"] == "a#1"
    assert d["text"] == "hello"
    ch2 = Chunk.from_dict(d)
    assert ch2.id == "a#1"
