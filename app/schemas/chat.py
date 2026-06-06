from pydantic import BaseModel, Field, field_validator

from app.services.ingestion import InvalidKnowledgeBaseError, validate_kb_id


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    kb_id: str = Field(min_length=1, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None

    @field_validator("kb_id")
    @classmethod
    def validate_kb_id_field(cls, value: str) -> str:
        try:
            return validate_kb_id(value)
        except InvalidKnowledgeBaseError as exc:
            raise ValueError(str(exc)) from exc


class SourceCitation(BaseModel):
    text: str
    score: float
    filename: str
    chunk_index: int
    doc_id: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
