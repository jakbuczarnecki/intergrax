# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_memcached_key_value_cache", "register_memcached_integration"]

def __getattr__(name: str):
    if name == "register_memcached_integration":
        from intergrax.integrations.providers.key_value_cache.memcached.register import register_memcached_integration
        return register_memcached_integration
    if name == "create_memcached_key_value_cache":
        from intergrax.integrations.providers.key_value_cache.memcached.bundle import create_memcached_key_value_cache
        return create_memcached_key_value_cache
    raise AttributeError(name)
