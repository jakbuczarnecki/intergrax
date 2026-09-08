# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox execution contracts shared by local and hosted sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from intergrax.runtime.sandbox.models import SandboxExecutionResult
from intergrax.runtime.sandbox.network_egress import NetworkEgressAllowlist

if TYPE_CHECKING:
    from intergrax.integrations.contracts.sandbox_host import SandboxSession

IsolationTierEvidence = Literal["local", "container", "cloud"]
NetworkEgressPolicy = Literal["deny", "allowlist"]


@dataclass(frozen=True, slots=True)
class SandboxNetworkEgressEvidence:
    """Trusted substrate network egress proof — actual enforcement, not requested config."""

    network_egress_deny_enforced: bool | None = None
    network_egress_allowlist_enforced: bool | None = None
    enforced_network_hosts: NetworkEgressAllowlist | None = None


@dataclass(frozen=True, slots=True)
class SandboxSecurityRequirements:
    """Typed security requirements for hosted substrate session creation."""

    isolation_tier: IsolationTierEvidence
    network_egress: NetworkEgressPolicy = "deny"
    network_egress_allowlist: NetworkEgressAllowlist | None = None


@dataclass(frozen=True, slots=True)
class SandboxSecurityCapabilities:
    """Trusted substrate security capability evidence (Sandbox-owned contract)."""

    isolation_tier: IsolationTierEvidence
    provider_id: str
    network_egress_deny_enforced: bool | None = None
    """``True`` when egress deny is proven; ``False`` when proven absent; ``None`` when unknown."""
    network_egress_allowlist_enforced: bool | None = None
    """``True`` when exact host allowlist is proven; ``False`` when proven absent; ``None`` when unknown."""
    enforced_network_hosts: NetworkEgressAllowlist | None = None
    """Hosts actually enforced by substrate — not requested profile scope."""

    @property
    def network_egress_evidence(self) -> SandboxNetworkEgressEvidence:
        return SandboxNetworkEgressEvidence(
            network_egress_deny_enforced=self.network_egress_deny_enforced,
            network_egress_allowlist_enforced=self.network_egress_allowlist_enforced,
            enforced_network_hosts=self.enforced_network_hosts,
        )


@runtime_checkable
class SandboxSecurityCapable(Protocol):
    """Sessions and optional host backends that attest security capabilities."""

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        """Return trusted substrate security capability evidence."""
        ...


@runtime_checkable
class SandboxSecurityConfigurable(Protocol):
    """Hosted backends that accept typed security requirements at session creation."""

    def create_session_with_security(
        self,
        requirements: SandboxSecurityRequirements,
    ) -> SandboxSession:
        """Provision a sandbox session with provider-enforced security configuration."""
        ...


@runtime_checkable
class SandboxSessionSecurityEvidenceProvider(Protocol):
    """Hosted backends that expose session-scoped security attestation evidence."""

    def session_security_capabilities(self, session_id: str) -> SandboxSecurityCapabilities:
        """Return trusted security evidence for one admitted sandbox session."""
        ...


@runtime_checkable
class SandboxExecCapable(Protocol):
    """Minimal surface required by the ``sandbox.exec`` catalog tool."""

    session_id: str

    def execute(self, operation: str, payload: dict | None = None) -> SandboxExecutionResult:
        """Run an allowlisted sandbox operation."""
        ...
