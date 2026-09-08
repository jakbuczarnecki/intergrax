# © Artur Czarnecki. All rights reserved.

"""Immutable, secret-free evidence contracts for E2B physical egress qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NetworkProbeResult:
    """Observed network probe outcome inside a sandbox session."""

    reachable: bool
    status_code: int | None
    redirected: bool


@dataclass(frozen=True, slots=True)
class HostProbeEvidence:
    """Single-host probe evidence collected from provider execution."""

    target: str
    result: NetworkProbeResult


@dataclass(frozen=True, slots=True)
class ProviderAttestationEvidence:
    """Provider-reported security attestation — not requested configuration."""

    provider_id: str
    network_egress_allowlist_enforced: bool | None
    enforced_network_hosts: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ControlPhaseEvidence:
    """C0 baseline — both hosts must be reachable without allowlist policy."""

    allowed_host: HostProbeEvidence
    denied_host: HostProbeEvidence
    baseline_valid: bool


@dataclass(frozen=True, slots=True)
class QualifiedPhaseEvidence:
    """C3 allowlist — declared host reachable, undeclared host blocked."""

    allowed_host: HostProbeEvidence
    denied_host: HostProbeEvidence
    provider_attestation: ProviderAttestationEvidence | None


@dataclass(frozen=True, slots=True)
class RedirectEvidence:
    """Redirect escape attempt from allowed host toward blocked destination."""

    attempted: bool
    escaped: bool
    redirect_url: str
    result: NetworkProbeResult


@dataclass(frozen=True, slots=True)
class RedirectPhaseEvidence:
    """Qualified redirect scenario phase wrapper."""

    redirect: RedirectEvidence


@dataclass(frozen=True, slots=True)
class CleanupEvidence:
    """Sandbox teardown outcome — failures do not hide phase failures."""

    session_id: str
    destroyed: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PhysicalEgressQualificationEvidence:
    """Complete causal-proof evidence bundle for one qualification scenario."""

    scenario_id: str
    provider: str
    control_phase: ControlPhaseEvidence
    qualified_phase: QualifiedPhaseEvidence
    redirect_phase: RedirectPhaseEvidence
    cleanup_phases: tuple[CleanupEvidence, ...]

    def to_mapping(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable evidence snapshot."""
        return asdict(self)

    def passes_causal_proof(self) -> bool:
        """True when all phases satisfy causal-proof expectations."""
        control = self.control_phase
        qualified = self.qualified_phase
        redirect = self.redirect_phase.redirect
        return (
            control.baseline_valid
            and control.allowed_host.result.reachable
            and control.denied_host.result.reachable
            and qualified.allowed_host.result.reachable
            and not qualified.denied_host.result.reachable
            and redirect.attempted
            and not redirect.escaped
        )
