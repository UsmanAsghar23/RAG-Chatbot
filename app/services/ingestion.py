import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache

from app.config import Settings, get_settings
from app.services.chunking import chunk_document
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.parsers import (
    DocumentParseError,
    EmptyDocumentError,
    parse_document,
)
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

INGEST_TIMEOUT_SECONDS = 120.0
KB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class InvalidKnowledgeBaseError(ValueError):
    """Raised when kb_id is invalid."""


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    kb_id: str
    filename: str
    source_type: str
    chunks_ingested: int


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    async def ingest_document(
        self,
        *,
        kb_id: str,
        filename: str,
        content: bytes,
    ) -> IngestResult:
        validated_kb_id = validate_kb_id(kb_id)
        doc_id = str(uuid.uuid4())

        document = parse_document(content, filename)
        chunks = chunk_document(
            doc_id=doc_id,
            segments=document.segments,
            source_type=document.source_type,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError(f"No chunks produced from {filename}")

        async def _run() -> int:
            embeddings = await self._embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            return await self._vector_store.upsert_chunks(
                kb_id=validated_kb_id,
                filename=filename,
                chunks=chunks,
                embeddings=embeddings,
            )

        try:
            chunks_ingested = await asyncio.wait_for(
                _run(),
                timeout=INGEST_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            logger.exception("Ingest timed out for %s", filename)
            raise TimeoutError("Ingest timed out while embedding or storing vectors") from exc

        return IngestResult(
            doc_id=doc_id,
            kb_id=validated_kb_id,
            filename=filename,
            source_type=document.source_type,
            chunks_ingested=chunks_ingested,
        )


def validate_kb_id(kb_id: str) -> str:
    normalized = kb_id.strip()
    if not KB_ID_PATTERN.fullmatch(normalized):
        raise InvalidKnowledgeBaseError(
            "kb_id must be 1-64 characters and contain only letters, numbers, hyphens, or underscores"
        )
    return normalized


@lru_cache
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        settings=settings,
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )
