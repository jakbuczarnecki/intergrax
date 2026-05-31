# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_oracle_relational_store", "register_oracle_integration"]

def __getattr__(name: str):
    if name == "register_oracle_integration":
        from intergrax.integrations.providers.relational_store.oracle.register import register_oracle_integration
        return register_oracle_integration
    if name == "create_oracle_relational_store":
        from intergrax.integrations.providers.relational_store.oracle.bundle import create_oracle_relational_store
        return create_oracle_relational_store
    raise AttributeError(name)
