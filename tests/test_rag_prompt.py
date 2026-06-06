from app.services.vector_store import ScoredChunk
from app.services.rag import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, _build_messages, _format_context


def test_build_messages_includes_context_and_question():
    chunks = [
        ScoredChunk(
            vector_id="doc-1_0",
            score=0.9,
            doc_id="doc-1",
            filename="notes.md",
            chunk_index=0,
            source_type="markdown",
            text="The capital of France is Paris.",
            page=None,
        )
    ]

    messages = _build_messages("What is the capital of France?", chunks)

    assert messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in messages[0]["content"]
    assert "The capital of France is Paris." in messages[1]["content"]
    assert "What is the capital of France?" in messages[1]["content"]
    assert "Answer using only the context above." in messages[1]["content"]


def test_format_context_labels_sources():
    chunks = [
        ScoredChunk(
            vector_id="doc-1_0",
            score=0.9,
            doc_id="doc-1",
            filename="guide.pdf",
            chunk_index=2,
            source_type="pdf",
            text="Sample text.",
            page=4,
        )
    ]

    context = _format_context(chunks)

    assert "[Source 1: guide.pdf, page 4, chunk 2]" in context
    assert "Sample text." in context


def test_no_context_answer_constant():
    assert "don't have enough information" in NO_CONTEXT_ANSWER.lower()
