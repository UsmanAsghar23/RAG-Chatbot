# RAG Chatbot

Production RAG chatbot: FastAPI, OpenAI embeddings, Pinecone, GPT-4.

Protected routes require header `X-API-Key: <API_KEY from .env>`.

Ingest a document:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: $API_KEY" \
  -F "kb_id=demo" \
  -F "file=@tests/fixtures/sample.md"
```

Expected response shape:

```json
{
  "doc_id": "...",
  "kb_id": "demo",
  "filename": "sample.md",
  "source_type": "markdown",
  "chunks_ingested": 3
}
```

```

Ask a question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "kb_id": "demo"
  }'
```

Streaming response:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "kb_id": "demo"
  }'
```

## Part 3 — Pinecone index setup

```bash
# Requires real PINECONE_API_KEY and OPENAI_API_KEY in .env
python scripts/ensure_pinecone_index.py
```

Index defaults: `rag-index`, dimension 1536, cosine metric, serverless AWS `us-east-1`.

## Part 1 — Local setup

```bash
cd /Users/usmanasghar23/CSProjects/Personal/RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with real API keys (dummy values work for /health only)
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

Protected routes (`/ingest`, `/chat` in later parts) will require header:

```bash
X-API-Key: <your API_KEY from .env>
```

## Implementation status

| Part | Status |
|------|--------|
| 1 — Scaffold | Done |
| 2 — Parsing & chunking | Done |
| 3 — Embeddings & Pinecone | Done |
| 4 — Ingest API | Done |
| 5 — Chat / RAG API | Done |
| 6 — Docker | Pending |
| 7 — AWS & CI | Pending |
