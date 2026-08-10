# Chroma (chroma)

Category: `vector_store`

## Single public entrypoint

- **`ChromaVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ChromaVectorStoreIntegration`.
- Contract factory: `create_chroma_vector_store_integration()`.

## Production runtime

Production configuration is HTTP/server-only by default:

```text
INTERGRAX_CHROMA_MODE=http
INTERGRAX_CHROMA_HOST=localhost
INTERGRAX_CHROMA_PORT=8000
```

The opener calls Chroma's `/api/v2/heartbeat` through `HttpClient` before the
collection is opened. Missing or unreachable server configuration fails closed;
the provider does not fall back to a local client.

`mode="embedded"` is an explicit development/test opt-in only. It is not
production or live-qualification evidence.

The repo-owned qualification service is:
`infra/docker/chromadb/docker-compose.yml`, using the pinned pair
`chromadb==1.4.1` and the `chromadb/chroma:1.4.1`-based qualification image.
Chroma remains
`QUALIFIED_OFFLINE_CONTRACT`; live qualification is pending
`RAG-LIVE-15B-R2`.
