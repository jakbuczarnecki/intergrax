# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_supabase_relational_store", "register_supabase_integration"]

def __getattr__(name: str):
    if name == "register_supabase_integration":
        from intergrax.integrations.providers.relational_store.supabase.register import register_supabase_integration
        return register_supabase_integration
    if name == "create_supabase_relational_store":
        from intergrax.integrations.providers.relational_store.supabase.bundle import create_supabase_relational_store
        return create_supabase_relational_store
    raise AttributeError(name)
