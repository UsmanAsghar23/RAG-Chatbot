from pydantic import BaseModel, Field

from app.services.parsers import SourceType


class IngestResponse(BaseModel):
    doc_id: str
    kb_id: str
    filename: str
    source_type: SourceType
    chunks_ingested: int = Field(ge=0)
