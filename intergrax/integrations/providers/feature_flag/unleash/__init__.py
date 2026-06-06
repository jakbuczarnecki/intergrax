# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_unleash_feature_flag", "register_unleash_integration"]

def __getattr__(name: str):
    if name == "register_unleash_integration":
        from intergrax.integrations.providers.feature_flag.unleash.register import register_unleash_integration
        return register_unleash_integration
    if name == "create_unleash_feature_flag":
        from intergrax.integrations.providers.feature_flag.unleash.bundle import create_unleash_feature_flag
        return create_unleash_feature_flag
    raise AttributeError(name)
