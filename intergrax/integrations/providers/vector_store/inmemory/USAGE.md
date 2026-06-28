# Inmemory (inmemory)

Category: `vector_store`

## Legacy facade

- `create_inmemory_vector_store()` remains backward-compatible.

## Contract-based integration

- `InmemoryVectorStoreIntegration` derives from the category-specific contract.
- Factory: `create_inmemory_vector_store_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
