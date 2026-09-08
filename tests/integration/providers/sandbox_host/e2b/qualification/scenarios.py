# © Artur Czarnecki. All rights reserved.

"""E2B physical egress qualification scenario definitions."""

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist


@dataclass(frozen=True, slots=True)
class E2bPhysicalEgressScenario:
    """Deterministic public endpoints for causal egress qualification."""

    scenario_id: str
    provider: str
    allowed_host: str
    denied_host: str
    redirect_url: str

    @property
    def allowlist(self):
        return canonicalize_network_egress_allowlist([self.allowed_host])


def default_e2b_physical_egress_scenario() -> E2bPhysicalEgressScenario:
    allowed = os.environ.get("INTERGRAX_E2B_QUAL_ALLOWED_HOST", "https://httpbin.org").rstrip("/")
    denied = os.environ.get("INTERGRAX_E2B_QUAL_DENIED_HOST", "https://www.google.com").rstrip("/")
    redirect = os.environ.get(
        "INTERGRAX_E2B_QUAL_REDIRECT_URL",
        "https://httpbin.org/redirect-to?url=https://www.google.com",
    )
    return E2bPhysicalEgressScenario(
        scenario_id="e2b-physical-egress-causal-proof",
        provider="e2b",
        allowed_host=allowed,
        denied_host=denied,
        redirect_url=redirect,
    )
