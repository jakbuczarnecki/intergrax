# Sqlite (sqlite)

Category: `relational_store`

## Legacy facade

- `create_sqlite_integration()` remains backward-compatible.

## Contract-based integration

- `SqliteRelationalStoreIntegration` derives from the category-specific contract.
- Factory: `create_sqlite_relational_store_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
