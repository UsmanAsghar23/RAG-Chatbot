import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("API_KEY", "test-api-key")


@pytest.fixture
def app():
    from app.main import app as fastapi_app

    return fastapi_app


@pytest.mark.anyio
async def test_openapi_lists_core_routes(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/ingest" in paths
    assert "/chat" in paths
    assert "/chat/stream" in paths


@pytest.mark.anyio
async def test_health_is_public(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_ingest_and_chat_require_api_key(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        ingest_response = await client.post("/ingest", data={"kb_id": "demo"})
        chat_response = await client.post(
            "/chat",
            json={"question": "Hello?", "kb_id": "demo"},
        )

    assert ingest_response.status_code == 401
    assert chat_response.status_code == 401


@pytest.mark.anyio
async def test_chat_rejects_invalid_payload(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/chat",
            headers={"X-API-Key": "test-api-key"},
            json={"question": "", "kb_id": "demo"},
        )

    assert response.status_code == 422
