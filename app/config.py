from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Chatbot"
    debug: bool = False

    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "rag-index"

    api_key: str

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    max_upload_mb: int = 10

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
