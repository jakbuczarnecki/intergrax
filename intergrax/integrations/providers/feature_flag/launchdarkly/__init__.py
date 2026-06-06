# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_launchdarkly_feature_flag", "register_launchdarkly_integration"]

def __getattr__(name: str):
    if name == "register_launchdarkly_integration":
        from intergrax.integrations.providers.feature_flag.launchdarkly.register import register_launchdarkly_integration
        return register_launchdarkly_integration
    if name == "create_launchdarkly_feature_flag":
        from intergrax.integrations.providers.feature_flag.launchdarkly.bundle import create_launchdarkly_feature_flag
        return create_launchdarkly_feature_flag
    raise AttributeError(name)
