# Qdrant (qdrant)

Category: `vector_store`

**Operator guide:** [`docs/project/technical/guides/RAG_OPERATOR_GUIDE.md`](../../../../../docs/project/technical/guides/RAG_OPERATOR_GUIDE.md)

Provider status: `QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED` (RAG-PROD-13).

## Single public entrypoint

- **`QdrantVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Contract factory: `create_qdrant_vector_store_integration()`.
- Legacy compatibility shims: `create_qdrant_vector_store()`, `create_qdrant_integration()`.
- Legacy shims construct the same `QdrantVectorStoreIntegration` via `from_store()` (inner RAG store).

## Runtime behavior

- Vector store operations (`add_documents`, `query`, `delete`, `count`) live on `QdrantVectorStoreIntegration`.
- Inner RAG store is accessed via `.rag_store`; catalog settings via `.store_config`.
- Qdrant client is imported only in `opens.py`.

## Contract path

- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in `integration.py`.
- Injectable `QdrantVectorStoreClient` required when `enabled=True`.

## Registry

- `register_qdrant_integration()` remains legacy-compatible (registers `create_qdrant_vector_store` shim).
- Registry v2 / contract registry wiring deferred.

## Removed

- Public `adapter.py` facade - behavior merged into `QdrantVectorStoreIntegration`.
