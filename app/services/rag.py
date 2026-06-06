import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.schemas.chat import SourceCitation
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.ingestion import validate_kb_id
from app.services.vector_store import ScoredChunk, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

CHAT_TIMEOUT_SECONDS = 60.0
CHAT_TEMPERATURE = 0.2

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the knowledge base to answer that question."
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context from a knowledge base.

Rules:
- Answer ONLY using information from the context blocks below.
- If the context does not contain enough information to answer the question, respond with exactly: "I don't have enough information in the knowledge base to answer that question."
- Do not use outside knowledge or invent facts.
- Be concise and accurate."""


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[SourceCitation]


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def retrieve_context(
        self,
        *,
        question: str,
        kb_id: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        return await self._retrieve(
            question=question,
            kb_id=kb_id,
            top_k=top_k,
            doc_ids=doc_ids,
        )

    async def chat(
        self,
        *,
        question: str,
        kb_id: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> ChatResult:
        chunks = await self._retrieve(
            question=question,
            kb_id=kb_id,
            top_k=top_k,
            doc_ids=doc_ids,
        )
        sources = to_sources(chunks)

        if not chunks:
            return ChatResult(answer=NO_CONTEXT_ANSWER, sources=[])

        messages = _build_messages(question, chunks)
        response = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                temperature=CHAT_TEMPERATURE,
            ),
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        answer = response.choices[0].message.content or NO_CONTEXT_ANSWER
        return ChatResult(answer=answer.strip(), sources=sources)

    async def chat_stream(
        self,
        *,
        question: str,
        kb_id: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        chunks: list[ScoredChunk] | None = None,
    ) -> AsyncIterator[str]:
        if chunks is None:
            chunks = await self._retrieve(
                question=question,
                kb_id=kb_id,
                top_k=top_k,
                doc_ids=doc_ids,
            )

        if not chunks:
            yield NO_CONTEXT_ANSWER
            return

        messages = _build_messages(question, chunks)
        stream = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                temperature=CHAT_TEMPERATURE,
                stream=True,
            ),
            timeout=CHAT_TIMEOUT_SECONDS,
        )

        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    async def _retrieve(
        self,
        *,
        question: str,
        kb_id: str,
        top_k: int | None,
        doc_ids: list[str] | None,
    ) -> list[ScoredChunk]:
        validated_kb_id = validate_kb_id(kb_id)
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        query_embedding = await self._embedding_service.embed_query(normalized_question)
        return await self._vector_store.query(
            kb_id=validated_kb_id,
            query_embedding=query_embedding,
            top_k=top_k or self._settings.top_k,
            doc_ids=doc_ids,
        )


def _build_messages(question: str, chunks: list[ScoredChunk]) -> list[dict[str, str]]:
    context = _format_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question.strip()}\n\n"
                "Answer using only the context above."
            ),
        },
    ]


def _format_context(chunks: list[ScoredChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        label = f"[Source {index}: {chunk.filename}"
        if chunk.page is not None:
            label += f", page {chunk.page}"
        label += f", chunk {chunk.chunk_index}]"
        blocks.append(f"{label}\n{chunk.text}")
    return "\n\n".join(blocks)


def to_sources(chunks: list[ScoredChunk]) -> list[SourceCitation]:
    return [
        SourceCitation(
            text=chunk.text,
            score=chunk.score,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            doc_id=chunk.doc_id,
            page=chunk.page,
        )
        for chunk in chunks
    ]


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    return RAGService(
        settings=settings,
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )
