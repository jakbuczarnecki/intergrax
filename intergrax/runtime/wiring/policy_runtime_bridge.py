# © Artur Czarnecki. All rights reserved.

"""Map composed ``RuntimePolicyBundle`` fields onto ``RuntimeConfig``."""

from __future__ import annotations

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle


def apply_policy_bundle_to_runtime_config(
    config: RuntimeConfig,
    bundle: RuntimePolicyBundle | None,
) -> RuntimeConfig:
    """Attach composed policy bundle; fill Nexus tool-scope slot when unset."""
    if bundle is None:
        return config
    config.policy_bundle = bundle
    if bundle.tool_access is not None and config.tool_scope_policy is None:
        config.tool_scope_policy = bundle.tool_access
    return config
