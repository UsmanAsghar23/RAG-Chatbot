import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from openai import APIError, APITimeoutError

from app.config import Settings, get_settings
from app.dependencies import verify_api_key
from app.schemas.ingest import IngestResponse
from app.services.ingestion import (
    IngestionService,
    InvalidKnowledgeBaseError,
    get_ingestion_service,
    validate_kb_id,
)
from app.services.parsers import DocumentParseError, EmptyDocumentError, get_source_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
async def ingest_document(
    kb_id: str = Form(...),
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    filename = _resolve_filename(file.filename)

    try:
        validate_kb_id(kb_id)
    except InvalidKnowledgeBaseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        get_source_type(filename)
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    content = await _read_upload(file, settings.max_upload_bytes)

    try:
        result = await ingestion_service.ingest_document(
            kb_id=kb_id,
            filename=filename,
            content=content,
        )
    except InvalidKnowledgeBaseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except (APITimeoutError, APIError) as exc:
        logger.exception("External API error during ingest for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to embed or store document vectors",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected ingest failure for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during document ingest",
        ) from exc

    return IngestResponse(
        doc_id=result.doc_id,
        kb_id=result.kb_id,
        filename=result.filename,
        source_type=result.source_type,
        chunks_ingested=result.chunks_ingested,
    )


def _resolve_filename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a filename",
        )
    return filename.strip()


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum size of {max_bytes // (1024 * 1024)} MB",
        )
    return content
