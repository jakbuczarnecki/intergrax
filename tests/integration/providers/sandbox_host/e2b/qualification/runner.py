# © Artur Czarnecki. All rights reserved.

"""E2B physical egress qualification runner — lifecycle orchestration only."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Iterator
from uuid import uuid4

from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.runtime.sandbox.contracts import (
    SandboxSecurityCapabilities,
    SandboxSecurityRequirements,
)
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession

from .errors import (
    QualificationAssertionError,
    QualificationBaselineError,
    QualificationSessionError,
)
from .models import (
    CleanupEvidence,
    ControlPhaseEvidence,
    HostProbeEvidence,
    PhysicalEgressQualificationEvidence,
    ProviderAttestationEvidence,
    QualifiedPhaseEvidence,
    RedirectEvidence,
    RedirectPhaseEvidence,
)
from .attestation_correlation import ProviderAttestationCorrelation
from .probes import HostedPythonNetworkProbe, SandboxNetworkProbe
from .scenarios import E2bPhysicalEgressScenario


class QualificationSandboxProvider:
    """Thin provider adapter — session open/destroy only, no policy authority."""

    def __init__(
        self,
        backend: SandboxHostBackend,
        *,
        tenant_id: str = "qual-tenant",
        task_id: str = "qual-task",
    ) -> None:
        self._backend = backend
        self._tenant_id = tenant_id
        self._task_id = task_id

    def open_session(
        self,
        security_requirements: SandboxSecurityRequirements | None,
    ) -> HostedSandboxSession:
        session = HostedSandboxSession.open(
            self._backend,
            tenant_id=self._tenant_id,
            task_id=self._task_id,
            security_requirements=security_requirements,
            allowed_operations=frozenset({"run_python"}),
        )
        if session is None:
            raise QualificationSessionError("hosted sandbox session was not admitted")
        return session

    def destroy_session(self, session_id: str) -> CleanupEvidence:
        error: str | None = None
        destroyed = False
        try:
            self._backend.destroy_session(session_id)
            destroyed = True
        except Exception as exc:  # noqa: BLE001 — qualification cleanup boundary
            error = f"{type(exc).__name__}: {exc}"
        return CleanupEvidence(session_id=session_id, destroyed=destroyed, error=error)


def _attestation_from_capabilities(
    capabilities: SandboxSecurityCapabilities | None,
) -> ProviderAttestationEvidence | None:
    if capabilities is None:
        return None
    enforced_hosts = capabilities.enforced_network_hosts
    host_tuple: tuple[str, ...] | None = None
    if enforced_hosts is not None:
        host_tuple = tuple(str(host) for host in enforced_hosts.hosts)
    return ProviderAttestationEvidence(
        provider_id=capabilities.provider_id,
        network_egress_allowlist_enforced=capabilities.network_egress_allowlist_enforced,
        enforced_network_hosts=host_tuple,
    )


class QualificationRunner:
    """Orchestrate causal-proof phases and aggregate immutable evidence."""

    def __init__(
        self,
        provider: QualificationSandboxProvider,
        scenario: E2bPhysicalEgressScenario,
        probe: SandboxNetworkProbe | None = None,
    ) -> None:
        self._provider = provider
        self._scenario = scenario
        self._probe = probe or HostedPythonNetworkProbe()

    @contextmanager
    def _phase_session(
        self,
        security_requirements: SandboxSecurityRequirements | None,
        cleanup_records: list[CleanupEvidence],
    ) -> Iterator[HostedSandboxSession]:
        session = self._provider.open_session(security_requirements)
        try:
            yield session
        finally:
            cleanup_records.append(self._provider.destroy_session(session.session_id))

    def run_control_phase(
        self,
        cleanup_records: list[CleanupEvidence],
    ) -> ControlPhaseEvidence:
        with self._phase_session(None, cleanup_records) as session:
            allowed = self._probe.execute(session, self._scenario.allowed_host)
            denied = self._probe.execute(session, self._scenario.denied_host)
        baseline_valid = allowed.reachable and denied.reachable
        return ControlPhaseEvidence(
            allowed_host=HostProbeEvidence(self._scenario.allowed_host, allowed),
            denied_host=HostProbeEvidence(self._scenario.denied_host, denied),
            baseline_valid=baseline_valid,
        )

    def run_qualified_phase(
        self,
        cleanup_records: list[CleanupEvidence],
    ) -> QualifiedPhaseEvidence:
        requirements = SandboxSecurityRequirements(
            isolation_tier="cloud",
            network_egress="allowlist",
            network_egress_allowlist=self._scenario.allowlist,
        )
        with self._phase_session(requirements, cleanup_records) as session:
            allowed = self._probe.execute(session, self._scenario.allowed_host)
            denied = self._probe.execute(session, self._scenario.denied_host)
            attestation = _attestation_from_capabilities(session.security_capabilities())
        allowed_evidence = HostProbeEvidence(self._scenario.allowed_host, allowed)
        denied_evidence = HostProbeEvidence(self._scenario.denied_host, denied)
        correlation = ProviderAttestationCorrelation.evaluate(
            requested_allowlist=self._scenario.allowlist,
            provider_attestation=attestation,
            allowed_probe=allowed_evidence,
            denied_probe=denied_evidence,
        )
        return QualifiedPhaseEvidence(
            allowed_host=allowed_evidence,
            denied_host=denied_evidence,
            provider_attestation=attestation,
            attestation_correlation=correlation,
        )

    def run_redirect_phase(
        self,
        cleanup_records: list[CleanupEvidence],
    ) -> RedirectPhaseEvidence:
        requirements = SandboxSecurityRequirements(
            isolation_tier="cloud",
            network_egress="allowlist",
            network_egress_allowlist=self._scenario.allowlist,
        )
        with self._phase_session(requirements, cleanup_records) as session:
            result = self._probe.execute(session, self._scenario.redirect_url)
        escaped = result.reachable and result.redirected
        return RedirectPhaseEvidence(
            redirect=RedirectEvidence(
                attempted=True,
                escaped=escaped,
                redirect_url=self._scenario.redirect_url,
                result=result,
            ),
        )

    def run(self) -> PhysicalEgressQualificationEvidence:
        cleanup_records: list[CleanupEvidence] = []
        execution_reference = f"{self._scenario.scenario_id}:{uuid4()}"
        timestamp_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
        control = self.run_control_phase(cleanup_records)
        qualified = self.run_qualified_phase(cleanup_records)
        redirect = self.run_redirect_phase(cleanup_records)
        return PhysicalEgressQualificationEvidence(
            scenario_id=self._scenario.scenario_id,
            provider_identity=self._scenario.provider,
            execution_reference=execution_reference,
            timestamp_utc=timestamp_utc,
            control_phase=control,
            qualified_phase=qualified,
            redirect_phase=redirect,
            cleanup_phases=tuple(cleanup_records),
        )

    def run_and_assert(self) -> PhysicalEgressQualificationEvidence:
        evidence = self.run()
        self.assert_causal_proof(evidence)
        return evidence

    @staticmethod
    def assert_causal_proof(evidence: PhysicalEgressQualificationEvidence) -> None:
        control = evidence.control_phase
        if not control.baseline_valid:
            raise QualificationBaselineError(
                "control phase denied host unreachable — no causal proof baseline",
            )
        qualified = evidence.qualified_phase
        if not qualified.allowed_host.result.reachable:
            raise QualificationAssertionError("qualified phase: allowed host not reachable")
        if qualified.denied_host.result.reachable:
            raise QualificationAssertionError("qualified phase: undeclared host reachable")
        redirect = evidence.redirect_phase.redirect
        if redirect.escaped:
            raise QualificationAssertionError("redirect phase: escape to blocked host succeeded")
        correlation = qualified.attestation_correlation
        if correlation is None or not correlation.passes():
            raise QualificationAssertionError(
                "attestation correlation: requested, attested, and observed scope mismatch",
            )
        for cleanup in evidence.cleanup_phases:
            if not cleanup.destroyed:
                raise QualificationAssertionError(
                    f"cleanup phase: sandbox session not destroyed ({cleanup.session_id})",
                )
