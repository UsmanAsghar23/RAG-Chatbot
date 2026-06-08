# Engineering Notes — RAG Service

The retrieval service uses OpenAI `text-embedding-3-small` embeddings stored in Pinecone.

Documents are chunked at 512 tokens with 64-token overlap. Each knowledge base is stored
in a separate Pinecone namespace identified by `kb_id`.

The chat endpoint retrieves the top five chunks by default and passes them as numbered
context blocks to GPT-4. Answers must be grounded in retrieved context only.

Health checks are exposed at `GET /health` for container and load balancer probes.
