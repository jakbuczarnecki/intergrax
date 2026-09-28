# © Artur Czarnecki. All rights reserved.

"""Deterministic physical egress qualification scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist


@dataclass(frozen=True, slots=True)
class PhysicalEgressScenario:
    """Controlled endpoints for causal egress qualification."""

    scenario_id: str
    provider: str
    allowed_host: str
    denied_host: str
    redirect_url: str

    @property
    def allowlist(self):
        return canonicalize_network_egress_allowlist([self.allowed_host])
