# Vespa (vespa)

Category: `vector_store`

## Legacy facade

- `create_vespa_integration()` remains backward-compatible.

## Contract-based integration

- `VespaVectorStoreIntegration` derives from the category-specific contract.
- Factory: `create_vespa_vector_store_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
