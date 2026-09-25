# Sqlite (sqlite)

Category: `relational_store`

## Single public entrypoint

- **`SqliteRelationalStoreIntegration`** in `integration.py` is the public provider integration class.
- Catalog factory: **`create_sqlite_relational_store()`** in `bundle.py` (also exported from the package root).
- Contract factory: **`create_sqlite_relational_store_integration()`** in `bundle.py`.
- Catalog registration: **`register_sqlite_integration()`** in `register.py`.
- Runtime domain-store composition: **`intergrax.runtime.persistence.sqlite_composition`** (not this package).
