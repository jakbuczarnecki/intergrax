# Jina Rerank (jina_rerank)

Category: `rerank_provider`

## Legacy facade

- `create_jina_rerank_provider()` remains backward-compatible.

## Contract-based integration

- `JinaRerankRerankProviderIntegration` derives from the category-specific contract.
- Factory: `create_jina_rerank_rerank_provider_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
