# Supabase (supabase)

Category: `relational_store`

## Single public entrypoint

- **`SupabaseRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SupabaseRelationalStoreIntegration`.
- Contract factory: `create_supabase_relational_store_integration()`.
