import os
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("API_KEY", "test-api-key")

from app.config import Settings
from app.services.chunking import TextChunk
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="rag-index",
        api_key="test-api-key",
    )


@pytest.fixture
def text_chunk() -> TextChunk:
    return TextChunk(
        chunk_id="abc123",
        text="The capital of France is Paris.",
        chunk_index=0,
        doc_id="doc-1",
        source_type="markdown",
        page=None,
    )


@pytest.mark.anyio
async def test_embed_texts_batches_and_returns_vectors(settings: Settings):
    service = EmbeddingService(settings, client=AsyncMock())
    service._client.embeddings.create = AsyncMock(
        return_value=MagicMock(
            data=[
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4]),
            ]
        )
    )

    vectors = await service.embed_texts(["one", "two"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2]
    service._client.embeddings.create.assert_awaited_once()


@pytest.mark.anyio
async def test_upsert_and_query_chunk(settings: Settings, text_chunk: TextChunk):
    mock_index = MagicMock()
    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    store = VectorStore(settings, client=mock_client)
    embedding = [0.01] * 1536

    count = await store.upsert_chunks(
        kb_id="demo",
        filename="notes.md",
        chunks=[text_chunk],
        embeddings=[embedding],
    )

    assert count == 1
    mock_index.upsert.assert_called_once()
    upsert_kwargs = mock_index.upsert.call_args.kwargs
    assert upsert_kwargs["namespace"] == "demo"
    vector = upsert_kwargs["vectors"][0]
    assert vector["id"] == "doc-1_0"
    assert vector["metadata"]["text"] == text_chunk.text
    assert vector["metadata"]["filename"] == "notes.md"

    mock_index.query.return_value = MagicMock(
        matches=[
            MagicMock(
                id="doc-1_0",
                score=0.92,
                metadata={
                    "doc_id": "doc-1",
                    "filename": "notes.md",
                    "chunk_index": 0,
                    "source_type": "markdown",
                    "text": text_chunk.text,
                },
            )
        ]
    )

    results = await store.query(
        kb_id="demo",
        query_embedding=embedding,
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].doc_id == "doc-1"
    assert results[0].score == pytest.approx(0.92)
    assert "Paris" in results[0].text


@pytest.mark.anyio
async def test_query_with_doc_id_filter(settings: Settings):
    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[])
    store = VectorStore(settings, client=MagicMock(Index=MagicMock(return_value=mock_index)))

    await store.query(
        kb_id="demo",
        query_embedding=[0.0] * 1536,
        top_k=5,
        doc_ids=["doc-1", "doc-2"],
    )

    assert mock_index.query.call_args.kwargs["filter"] == {
        "doc_id": {"$in": ["doc-1", "doc-2"]}
    }


@pytest.mark.anyio
async def test_ensure_index_skips_when_exists(settings: Settings):
    mock_client = MagicMock()
    existing = MagicMock()
    existing.name = "rag-index"
    mock_client.list_indexes.return_value = [existing]

    store = VectorStore(settings, client=mock_client)
    await store.ensure_index()

    mock_client.create_index.assert_not_called()


@pytest.mark.anyio
async def test_ensure_index_creates_when_missing(settings: Settings):
    mock_client = MagicMock()
    mock_client.list_indexes.return_value = []

    store = VectorStore(settings, client=mock_client)
    await store.ensure_index(cloud="aws", region="us-east-1")

    mock_client.create_index.assert_called_once()
    kwargs = mock_client.create_index.call_args.kwargs
    assert kwargs["name"] == "rag-index"
    assert kwargs["dimension"] == 1536
    assert kwargs["metric"] == "cosine"


@pytest.mark.anyio
async def test_integration_upsert_then_query_with_mocked_openai(
    settings: Settings,
    text_chunk: TextChunk,
):
    mock_index = MagicMock()
    stored_vectors: list[dict] = []

    def upsert(**kwargs):
        stored_vectors.extend(kwargs["vectors"])

    def query(**kwargs):
        return MagicMock(
            matches=[
                MagicMock(
                    id=stored_vectors[0]["id"],
                    score=0.99,
                    metadata=stored_vectors[0]["metadata"],
                )
            ]
        )

    mock_index.upsert.side_effect = upsert
    mock_index.query.side_effect = query

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    embedding_service = EmbeddingService(settings, client=AsyncMock())
    embedding_service._client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.5] * 1536)])
    )

    store = VectorStore(settings, client=mock_client)

    query_text = "What is the capital of France?"
    query_embedding = await embedding_service.embed_query(query_text)
    doc_embedding = (await embedding_service.embed_texts([text_chunk.text]))[0]

    await store.upsert_chunks(
        kb_id="demo",
        filename="notes.md",
        chunks=[text_chunk],
        embeddings=[doc_embedding],
    )

    results = await store.query(
        kb_id="demo",
        query_embedding=query_embedding,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].text == text_chunk.text
