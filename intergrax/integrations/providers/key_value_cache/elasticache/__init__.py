# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_elasticache_key_value_cache", "register_elasticache_integration"]

def __getattr__(name: str):
    if name == "register_elasticache_integration":
        from intergrax.integrations.providers.key_value_cache.elasticache.register import register_elasticache_integration
        return register_elasticache_integration
    if name == "create_elasticache_key_value_cache":
        from intergrax.integrations.providers.key_value_cache.elasticache.bundle import create_elasticache_key_value_cache
        return create_elasticache_key_value_cache
    raise AttributeError(name)
