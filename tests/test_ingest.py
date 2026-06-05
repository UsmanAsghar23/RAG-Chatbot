import os
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("API_KEY", "test-api-key")

from app.main import app
from app.services.ingestion import IngestResult, IngestionService, get_ingestion_service

FIXTURES = Path(__file__).parent / "fixtures"
API_KEY = "test-api-key"


@pytest.fixture
def mock_ingestion_service() -> AsyncMock:
    service = AsyncMock(spec=IngestionService)
    service.ingest_document = AsyncMock(
        return_value=IngestResult(
            doc_id="doc-123",
            kb_id="demo",
            filename="sample.md",
            source_type="markdown",
            chunks_ingested=3,
        )
    )
    return service


@pytest.fixture
def client(mock_ingestion_service: AsyncMock):
    app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion_service
    yield
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_ingest_success(client: None, mock_ingestion_service: AsyncMock):
    sample = (FIXTURES / "sample.md").read_bytes()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/ingest",
            headers={"X-API-Key": API_KEY},
            data={"kb_id": "demo"},
            files={"file": ("sample.md", BytesIO(sample), "text/markdown")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == "doc-123"
    assert body["kb_id"] == "demo"
    assert body["filename"] == "sample.md"
    assert body["source_type"] == "markdown"
    assert body["chunks_ingested"] == 3
    mock_ingestion_service.ingest_document.assert_awaited_once()


@pytest.mark.anyio
async def test_ingest_requires_api_key(client: None):
    sample = (FIXTURES / "sample.md").read_bytes()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/ingest",
            data={"kb_id": "demo"},
            files={"file": ("sample.md", BytesIO(sample), "text/markdown")},
        )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_ingest_rejects_unsupported_file(client: None):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/ingest",
            headers={"X-API-Key": API_KEY},
            data={"kb_id": "demo"},
            files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.anyio
async def test_ingest_rejects_empty_file(client: None):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/ingest",
            headers={"X-API-Key": API_KEY},
            data={"kb_id": "demo"},
            files={"file": ("sample.md", BytesIO(b""), "text/markdown")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


@pytest.mark.anyio
async def test_ingest_rejects_invalid_kb_id(client: None):
    sample = (FIXTURES / "sample.md").read_bytes()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        response = await http_client.post(
            "/ingest",
            headers={"X-API-Key": API_KEY},
            data={"kb_id": "bad id!"},
            files={"file": ("sample.md", BytesIO(sample), "text/markdown")},
        )

    assert response.status_code == 400
    assert "kb_id" in response.json()["detail"]


@pytest.mark.anyio
async def test_ingest_rejects_oversized_file(client: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from app.config import get_settings

    get_settings.cache_clear()

    oversized = b"x" * (1024 * 1024 + 1)

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as http_client:
            response = await http_client.post(
                "/ingest",
                headers={"X-API-Key": API_KEY},
                data={"kb_id": "demo"},
                files={"file": ("sample.md", BytesIO(oversized), "text/markdown")},
            )

        assert response.status_code == 413
    finally:
        get_settings.cache_clear()


@pytest.mark.anyio
async def test_ingestion_service_end_to_end_with_mocks():
    from app.config import Settings
    from app.services.ingestion import IngestionService

    settings = Settings(
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="rag-index",
        api_key="test-api-key",
        chunk_size=128,
        chunk_overlap=16,
    )

    embedding_service = AsyncMock()
    embedding_service.embed_texts = AsyncMock(
        side_effect=lambda texts: [[0.1] * 1536 for _ in texts]
    )

    vector_store = AsyncMock()
    vector_store.upsert_chunks = AsyncMock(side_effect=lambda **kwargs: len(kwargs["chunks"]))

    service = IngestionService(settings, embedding_service, vector_store)
    content = (FIXTURES / "sample.md").read_bytes()

    result = await service.ingest_document(
        kb_id="demo",
        filename="sample.md",
        content=content,
    )

    assert result.kb_id == "demo"
    assert result.filename == "sample.md"
    assert result.source_type == "markdown"
    assert result.chunks_ingested > 0
    embedding_service.embed_texts.assert_awaited_once()
    vector_store.upsert_chunks.assert_awaited_once()
