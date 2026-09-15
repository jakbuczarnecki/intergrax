# © Artur Czarnecki. All rights reserved.

"""Nexus adapter: ``RuntimeConfig`` → neutral tool policy resolution (GR-4-R1)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.policy.tool_policy_resolution import resolve_allowed_tools


def resolve_allowed_tools_from_runtime_config(
    config: RuntimeConfig,
    *,
    explicit: Sequence[str] | None = None,
) -> Sequence[str] | None:
    """Translate Nexus ``RuntimeConfig`` into neutral ``resolve_allowed_tools`` inputs."""
    bundle = config.policy_bundle
    upstream = bundle.tool_access if bundle is not None else None
    return resolve_allowed_tools(upstream_policy=upstream, explicit=explicit)
