# Semantic Scholar (semantic_scholar)

Category: `search_provider`

## Legacy facade

- `create_semantic_scholar_search_provider()` remains backward-compatible.

## Contract-based integration

- `SemanticScholarSearchProviderIntegration` derives from the category-specific contract.
- Factory: `create_semantic_scholar_search_provider_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
