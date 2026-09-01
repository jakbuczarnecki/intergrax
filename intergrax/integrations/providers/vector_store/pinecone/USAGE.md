# Pinecone (pinecone)

Category: `vector_store`

## Single public entrypoint

- **`PineconeVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Contract factory: `create_pinecone_vector_store_integration()`.
- Legacy compatibility shims: `create_pinecone_vector_store()`, `create_pinecone_integration()`.
- Legacy shims construct the same `PineconeVectorStoreIntegration` via `from_store()` (inner RAG store).

## Runtime behavior

- Vector store operations (`add_documents`, `query`, `delete`, `count`) live on `PineconeVectorStoreIntegration`.
- Inner RAG store is accessed via `.rag_store`; catalog settings via `.store_config`.
- Pinecone SDK is imported only in `opens.py`.

## Contract path

- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in `integration.py`.
- Injectable `PineconeVectorStoreClient` required when `enabled=True`.

## Registry

- `register_pinecone_integration()` remains legacy-compatible (registers `create_pinecone_vector_store` shim).
- Registry v2 / contract registry wiring deferred.

## Removed

- Public `adapter.py` facade - behavior merged into `PineconeVectorStoreIntegration`.
