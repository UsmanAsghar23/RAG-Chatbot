import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import APIError, APITimeoutError

from app.dependencies import verify_api_key
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ingestion import InvalidKnowledgeBaseError
from app.services.rag import RAGService, get_rag_service, to_sources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(
    payload: ChatRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatResponse:
    try:
        result = await rag_service.chat(
            question=payload.question,
            kb_id=payload.kb_id,
            top_k=payload.top_k,
            doc_ids=payload.doc_ids,
        )
    except InvalidKnowledgeBaseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Chat request timed out",
        ) from exc
    except (APITimeoutError, APIError) as exc:
        logger.exception("External API error during chat")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate chat response",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected chat failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during chat",
        ) from exc

    return ChatResponse(answer=result.answer, sources=result.sources)


@router.post("/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(
    payload: ChatRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> StreamingResponse:
    try:
        chunks = await rag_service.retrieve_context(
            question=payload.question,
            kb_id=payload.kb_id,
            top_k=payload.top_k,
            doc_ids=payload.doc_ids,
        )
    except InvalidKnowledgeBaseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (APITimeoutError, APIError) as exc:
        logger.exception("External API error during chat stream setup")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to retrieve context for chat stream",
        ) from exc

    sources = to_sources(chunks)

    async def event_generator():
        metadata = {
            "type": "sources",
            "sources": [source.model_dump() for source in sources],
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        try:
            async for token in rag_service.chat_stream(
                question=payload.question,
                kb_id=payload.kb_id,
                top_k=payload.top_k,
                doc_ids=payload.doc_ids,
                chunks=chunks,
            ):
                event_payload = {"type": "token", "content": token}
                yield f"data: {json.dumps(event_payload)}\n\n"
        except TimeoutError:
            event_payload = {"type": "error", "message": "Chat request timed out"}
            yield f"data: {json.dumps(event_payload)}\n\n"
        except (APITimeoutError, APIError):
            event_payload = {"type": "error", "message": "Failed to generate chat response"}
            yield f"data: {json.dumps(event_payload)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
