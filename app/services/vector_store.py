import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from pinecone import Pinecone, ServerlessSpec

from app.config import Settings, get_settings
from app.services.chunking import TextChunk
from app.services.parsers import SourceType

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536
METADATA_TEXT_MAX_CHARS = 1000
UPSERT_BATCH_SIZE = 100


@dataclass(frozen=True)
class ScoredChunk:
    vector_id: str
    score: float
    doc_id: str
    filename: str
    chunk_index: int
    source_type: SourceType
    text: str
    page: int | None = None


class VectorStore:
    def __init__(
        self,
        settings: Settings,
        client: Pinecone | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or Pinecone(api_key=settings.pinecone_api_key)
        self._index = None

    @property
    def index_name(self) -> str:
        return self._settings.pinecone_index_name

    def _get_index(self):
        if self._index is None:
            self._index = self._client.Index(self._settings.pinecone_index_name)
        return self._index

    async def ensure_index(
        self,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        await asyncio.to_thread(self._ensure_index_sync, cloud, region)

    def _ensure_index_sync(self, cloud: str, region: str) -> None:
        existing = {index.name for index in self._client.list_indexes()}
        if self.index_name in existing:
            return

        logger.info("Creating Pinecone index %s", self.index_name)
        self._client.create_index(
            name=self.index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    async def upsert_chunks(
        self,
        *,
        kb_id: str,
        filename: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length must match")
        if not chunks:
            return 0

        vectors = [
            {
                "id": _vector_id(chunk.doc_id, chunk.chunk_index),
                "values": embedding,
                "metadata": _build_metadata(chunk, filename),
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[start : start + UPSERT_BATCH_SIZE]
            await asyncio.to_thread(
                self._get_index().upsert,
                vectors=batch,
                namespace=kb_id,
            )

        return len(vectors)

    async def query(
        self,
        *,
        kb_id: str,
        query_embedding: list[float],
        top_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        filter_metadata: dict[str, Any] | None = None
        if doc_ids:
            filter_metadata = {"doc_id": {"$in": doc_ids}}

        response = await asyncio.to_thread(
            self._get_index().query,
            vector=query_embedding,
            top_k=top_k,
            namespace=kb_id,
            include_metadata=True,
            filter=filter_metadata,
        )

        results: list[ScoredChunk] = []
        for match in response.matches or []:
            metadata = match.metadata or {}
            results.append(
                ScoredChunk(
                    vector_id=match.id,
                    score=float(match.score or 0.0),
                    doc_id=str(metadata.get("doc_id", "")),
                    filename=str(metadata.get("filename", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    source_type=cast(SourceType, metadata.get("source_type", "markdown")),
                    text=str(metadata.get("text", "")),
                    page=_optional_int(metadata.get("page")),
                )
            )

        return results

    async def delete_document(self, *, kb_id: str, doc_id: str) -> None:
        await asyncio.to_thread(
            self._get_index().delete,
            filter={"doc_id": doc_id},
            namespace=kb_id,
        )


def _vector_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}_{chunk_index}"


def _build_metadata(chunk: TextChunk, filename: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "doc_id": chunk.doc_id,
        "filename": filename,
        "chunk_index": chunk.chunk_index,
        "source_type": chunk.source_type,
        "text": _truncate_text(chunk.text),
    }
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


def _truncate_text(text: str) -> str:
    if len(text) <= METADATA_TEXT_MAX_CHARS:
        return text
    return text[:METADATA_TEXT_MAX_CHARS]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings())
