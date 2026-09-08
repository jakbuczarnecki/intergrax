# © Artur Czarnecki. All rights reserved.

"""Immutable, secret-free evidence contracts for E2B physical egress qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class NetworkProbeResult:
    """Observed network probe outcome inside a sandbox session."""

    reachable: bool
    status_code: int | None
    redirect_target: str | None
    latency_ms: float | None
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
class ObservedNetworkScope:
    """Reachability observations used to verify enforcement against policy."""

    allowed_target: str
    allowed_reachable: bool
    denied_target: str
    denied_reachable: bool


@dataclass(frozen=True, slots=True)
class ProviderAttestationCorrelationEvidence:
    """Immutable correlation of requested, attested, and observed network scope."""

    requested_scope: tuple[str, ...]
    attested_scope: tuple[str, ...] | None
    observed_scope: ObservedNetworkScope
    attestation_verified: bool
    execution_verified: bool
    correlation_result: Literal["PASS", "DENIED"]

    def passes(self) -> bool:
        """True when requested, attested, and observed scopes align."""
        return self.correlation_result == "PASS"

    def to_mapping(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable evidence snapshot."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualifiedPhaseEvidence:
    """C3 allowlist — declared host reachable, undeclared host blocked."""

    allowed_host: HostProbeEvidence
    denied_host: HostProbeEvidence
    provider_attestation: ProviderAttestationEvidence | None
    attestation_correlation: ProviderAttestationCorrelationEvidence | None = None


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
    provider_identity: str
    execution_reference: str
    timestamp_utc: str
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
        correlation = qualified.attestation_correlation
        correlation_ok = correlation is None or correlation.passes()
        return (
            control.baseline_valid
            and control.allowed_host.result.reachable
            and control.denied_host.result.reachable
            and qualified.allowed_host.result.reachable
            and not qualified.denied_host.result.reachable
            and redirect.attempted
            and not redirect.escaped
            and correlation_ok
        )
