# RAG Chatbot

Production RAG chatbot: FastAPI, OpenAI embeddings, Pinecone, GPT-4.

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
| 3 — Embeddings & Pinecone | Pending |
| 4 — Ingest API | Pending |
| 5 — Chat / RAG API | Pending |
| 6 — Docker | Pending |
| 7 — AWS & CI | Pending |
