import os

import pytest
from httpx import ASGITransport, AsyncClient

# Minimal env so Settings loads during tests
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("API_KEY", "test-api-key")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_returns_ok():
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
