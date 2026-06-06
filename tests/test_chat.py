import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("API_KEY", "test-api-key")

from app.main import app
from app.schemas.chat import SourceCitation
from app.services.rag import ChatResult, NO_CONTEXT_ANSWER, RAGService, get_rag_service
from app.services.vector_store import ScoredChunk

API_KEY = "test-api-key"


@pytest.fixture
def mock_rag_service() -> AsyncMock:
    service = AsyncMock(spec=RAGService)
    service.chat = AsyncMock(
        return_value=ChatResult(
            answer="The capital of France is Paris.",
            sources=[
                SourceCitation(
                    text="The capital of France is Paris.",
                    score=0.92,
                    filename="sample.md",
                    chunk_index=0,
                    doc_id="doc-1",
                )
            ],
        )
    )
    service.retrieve_context = AsyncMock(return_value=[])
    async def stream_tokens(**kwargs):
        yield "The capital"
        yield " of France is Paris."

    service.chat_stream = stream_tokens
    return service


@pytest.fixture
def client(mock_rag_service: AsyncMock):
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    yield
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_chat_success(client: None, mock_rag_service: AsyncMock):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/chat",
            headers={"X-API-Key": API_KEY},
            json={
                "question": "What is the capital of France?",
                "kb_id": "demo",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The capital of France is Paris."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["filename"] == "sample.md"
    mock_rag_service.chat.assert_awaited_once()


@pytest.mark.anyio
async def test_chat_requires_api_key(client: None):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/chat",
            json={"question": "Hello?", "kb_id": "demo"},
        )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_chat_rejects_invalid_kb_id(client: None):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/chat",
            headers={"X-API-Key": API_KEY},
            json={"question": "Hello?", "kb_id": "bad id"},
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_chat_stream_returns_sse(client: None, mock_rag_service: AsyncMock):
    mock_rag_service.retrieve_context.return_value = [
        ScoredChunk(
            vector_id="doc-1_0",
            score=0.9,
            doc_id="doc-1",
            filename="sample.md",
            chunk_index=0,
            source_type="markdown",
            text="The capital of France is Paris.",
            page=None,
        )
    ]

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/chat/stream",
            headers={"X-API-Key": API_KEY},
            json={
                "question": "What is the capital of France?",
                "kb_id": "demo",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "sources"' in response.text
    assert '"type": "token"' in response.text
    assert '"type": "done"' in response.text


@pytest.mark.anyio
async def test_rag_service_no_context_skips_llm():
    from app.config import Settings

    settings = Settings(
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="rag-index",
        api_key="test-api-key",
    )

    embedding_service = AsyncMock()
    embedding_service.embed_query = AsyncMock(return_value=[0.1] * 1536)

    vector_store = AsyncMock()
    vector_store.query = AsyncMock(return_value=[])

    openai_client = AsyncMock()

    service = RAGService(settings, embedding_service, vector_store, client=openai_client)
    result = await service.chat(
        question="What is the capital of France?",
        kb_id="demo",
    )

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    openai_client.chat.completions.create.assert_not_called()


@pytest.mark.anyio
async def test_rag_service_chat_with_context():
    from app.config import Settings
    from app.services.vector_store import ScoredChunk

    settings = Settings(
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="rag-index",
        api_key="test-api-key",
    )

    chunk = ScoredChunk(
        vector_id="doc-1_0",
        score=0.95,
        doc_id="doc-1",
        filename="sample.md",
        chunk_index=0,
        source_type="markdown",
        text="The capital of France is Paris.",
    )

    embedding_service = AsyncMock()
    embedding_service.embed_query = AsyncMock(return_value=[0.1] * 1536)

    vector_store = AsyncMock()
    vector_store.query = AsyncMock(return_value=[chunk])

    openai_client = AsyncMock()
    openai_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Paris is the capital of France."))]
        )
    )

    service = RAGService(settings, embedding_service, vector_store, client=openai_client)
    result = await service.chat(
        question="What is the capital of France?",
        kb_id="demo",
    )

    assert "Paris" in result.answer
    assert len(result.sources) == 1
    openai_client.chat.completions.create.assert_awaited_once()
