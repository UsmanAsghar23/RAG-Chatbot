import asyncio
import logging
from functools import lru_cache

from openai import AsyncOpenAI, RateLimitError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 1.0


class EmbeddingService:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.embeddings.create(
                    model=self._settings.embedding_model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "OpenAI rate limit hit; retrying in %.1fs (attempt %d)",
                    delay,
                    attempt + 1,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("unreachable")


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())
