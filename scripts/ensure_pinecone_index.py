#!/usr/bin/env python3
"""Create the Pinecone index if it does not exist."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vector_store import get_vector_store


async def main() -> None:
    store = get_vector_store()
    await store.ensure_index()
    print(f"Pinecone index ready: {store.index_name}")


if __name__ == "__main__":
    asyncio.run(main())
