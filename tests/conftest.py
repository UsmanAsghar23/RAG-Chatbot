import os
from pathlib import Path

import pytest
import tiktoken

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("API_KEY", "test-api-key")

CACHE_DIR = Path(__file__).parent / ".tiktoken_cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(CACHE_DIR))


@pytest.fixture(scope="session", autouse=True)
def _warm_tiktoken_cache() -> None:
    tiktoken.get_encoding("cl100k_base")
