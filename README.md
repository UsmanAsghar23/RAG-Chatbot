# RAG Chatbot

Production RAG chatbot: FastAPI, OpenAI embeddings, Pinecone, GPT-4.

## Architecture

```text
Client
  │
  ▼
FastAPI (ingest + chat)
  │
  ├─ Ingest: parse PDF/Markdown → chunk → embed → Pinecone upsert
  └─ Chat:   embed question → Pinecone query → GPT-4 grounded answer
```

| Component | Role |
|-----------|------|
| FastAPI | Async HTTP API for ingest and chat |
| OpenAI | Embeddings (`text-embedding-3-small`) and chat (`gpt-4o`) |
| Pinecone | Vector storage and semantic search, namespace per `kb_id` |
| pypdf / tiktoken | PDF parsing and token-aware chunking |

## Quick start with Docker

```bash
cd /Users/usmanasghar23/CSProjects/Personal/RAG
cp .env.example .env
# Edit .env with real OPENAI_API_KEY, PINECONE_API_KEY, and API_KEY

docker compose up --build
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

### Pinecone index (first-time setup)

Run once with valid credentials before ingesting documents:

```bash
source .venv/bin/activate   # or use your local Python env
pip install -e .
python scripts/ensure_pinecone_index.py
```

Index defaults: `rag-index`, dimension 1536, cosine metric, serverless AWS `us-east-1`.

### End-to-end demo

```bash
export API_KEY=your-api-key-from-env

# Ingest sample docs into namespace "demo"
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: $API_KEY" \
  -F "kb_id=demo" \
  -F "file=@samples/product-faq.md"

curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: $API_KEY" \
  -F "kb_id=demo" \
  -F "file=@samples/company-policy.md"

# Ask a question grounded in the knowledge base
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How much does the Starter plan cost?",
    "kb_id": "demo"
  }'
```

Streaming chat:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the core collaboration hours for remote work?",
    "kb_id": "demo"
  }'
```

## Local development (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest
```

## API reference

Protected routes require `X-API-Key: <API_KEY>`.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe |
| `POST /ingest` | Upload PDF or Markdown (`kb_id`, `file`) |
| `POST /chat` | Grounded Q&A with `sources[]` |
| `POST /chat/stream` | SSE stream (`sources`, `token`, `done` events) |

## Implementation status

| Part | Status |
|------|--------|
| 1 — Scaffold | Done |
| 2 — Parsing & chunking | Done |
| 3 — Embeddings & Pinecone | Done |
| 4 — Ingest API | Done |
| 5 — Chat / RAG API | Done |
| 6 — Docker | Done |
| 7 — AWS & CI | Pending |
