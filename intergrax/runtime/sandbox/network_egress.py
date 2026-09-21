# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export — canonical: contracts.sandbox_network_egress."""

from __future__ import annotations

from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
    NetworkEgressScheme,
    NetworkEgressScopeError,
    allowlist_enforcement_satisfies_request,
    canonicalize_network_egress_allowlist,
    parse_network_egress_host,
)

__all__ = [
    "NetworkEgressAllowlist",
    "NetworkEgressHost",
    "NetworkEgressScheme",
    "NetworkEgressScopeError",
    "allowlist_enforcement_satisfies_request",
    "canonicalize_network_egress_allowlist",
    "parse_network_egress_host",
]
