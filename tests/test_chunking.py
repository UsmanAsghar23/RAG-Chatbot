import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("API_KEY", "test-api-key")

from app.services.chunking import chunk_document, chunk_text
from app.services.parsers import (
    EmptyDocumentError,
    parse_document,
    parse_markdown,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_markdown_strips_front_matter():
    content = b"---\ntitle: x\n---\n\n# Hello\n\nWorld."
    segments = parse_markdown(content)

    assert len(segments) == 1
    assert segments[0].text.startswith("# Hello")
    assert "title: x" not in segments[0].text


def test_parse_document_from_sample_md():
    content = (FIXTURES / "sample.md").read_bytes()
    document = parse_document(content, "sample.md")

    assert document.source_type == "markdown"
    assert len(document.segments) == 1
    assert "Paris" in document.segments[0].text


def test_parse_document_empty_markdown_raises():
    with pytest.raises(EmptyDocumentError):
        parse_document(b"   \n  ", "empty.md")


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_respects_paragraph_boundaries():
    paragraph = (
        "Paragraph one with enough words to exceed a small token budget when repeated."
    )
    text = "\n\n".join([paragraph] * 4)
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=5)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.strip()


def test_chunk_text_overlap_carries_content():
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 2
    first_tail = chunks[0].split()[-5:]
    second_head = chunks[1].split()[:5]
    assert first_tail[-1] == second_head[0]


def test_chunk_document_produces_stable_ids():
    content = (FIXTURES / "sample.md").read_bytes()
    document = parse_document(content, "sample.md")

    chunks_a = chunk_document(
        doc_id="doc-1",
        segments=document.segments,
        source_type=document.source_type,
        chunk_size=128,
        chunk_overlap=16,
    )
    chunks_b = chunk_document(
        doc_id="doc-1",
        segments=document.segments,
        source_type=document.source_type,
        chunk_size=128,
        chunk_overlap=16,
    )

    assert len(chunks_a) > 0
    assert [chunk.chunk_id for chunk in chunks_a] == [
        chunk.chunk_id for chunk in chunks_b
    ]
    assert chunks_a[0].chunk_index == 0
    assert "Paris" in " ".join(chunk.text for chunk in chunks_a)


def test_chunk_document_from_pdf_fixture():
    pdf_bytes = (FIXTURES / "sample.pdf").read_bytes()
    document = parse_document(pdf_bytes, "sample.pdf")
    chunks = chunk_document(
        doc_id="pdf-doc",
        segments=document.segments,
        source_type=document.source_type,
    )

    assert len(chunks) >= 1
    assert chunks[0].page == 1
    assert "Berlin" in chunks[0].text


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(
            doc_id="doc",
            segments=parse_markdown(b"hello world"),
            source_type="markdown",
            chunk_size=64,
            chunk_overlap=64,
        )
