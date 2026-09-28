# © Artur Czarnecki. All rights reserved.

"""Provider-neutral reference substrate physical egress qualification."""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path

import pytest

from intergrax.runtime.sandbox.contracts import SandboxSecurityRequirements
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from tests.integration.providers.sandbox_host.qualification import (
    NetworkProbeResult,
    QualificationRunner,
    QualificationSandboxProvider,
)
from tests.integration.runtime.sandbox.reference_substrate.backend import (
    ReferenceSandboxBackend,
    ReferenceSubstrateSecurityError,
    build_reference_backend_or_skip_reason,
)
from tests.integration.runtime.sandbox.reference_substrate.endpoints import (
    default_reference_scenario,
)
from tests.integration.runtime.sandbox.reference_substrate.firewall import (
    ReferenceSubstratePolicyError,
)
from tests.integration.runtime.sandbox.reference_substrate.preflight import (
    evaluate_reference_substrate_preflight,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.qualification,
]

_EVIDENCE_ROOT = Path(".tmp/session/reference-physical-egress-qualification")
_SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}"),
)


def _require_linux_physical_environment() -> ReferenceSandboxBackend:
    if platform.system() != "Linux":
        pytest.skip("reference substrate physical qualification requires Linux/WSL2")
    backend_or_reason = build_reference_backend_or_skip_reason()
    if isinstance(backend_or_reason, str):
        pytest.skip(backend_or_reason)
    return backend_or_reason


@pytest.fixture(scope="module")
def reference_backend() -> ReferenceSandboxBackend:
    return _require_linux_physical_environment()


@pytest.fixture(scope="module")
def qualification_runner(reference_backend: ReferenceSandboxBackend) -> QualificationRunner:
    provider = QualificationSandboxProvider(reference_backend)
    return QualificationRunner(provider, default_reference_scenario())


def test_preflight_contract_non_linux_reports_block_reason() -> None:
    preflight = evaluate_reference_substrate_preflight()
    if platform.system() == "Linux":
        assert preflight.linux_kernel
    else:
        assert preflight.ok is False
        assert preflight.block_reason is not None


def test_control_environment_proves_baseline_connectivity(
    qualification_runner: QualificationRunner,
) -> None:
    cleanup: list = []
    control = qualification_runner.run_control_phase(cleanup)
    assert control.baseline_valid is True
    assert control.allowed_host.result.reachable is True
    assert control.denied_host.result.reachable is True
    assert cleanup[0].destroyed is True


def test_allowlisted_host_is_reachable(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    qualified = qualification_runner.run_qualified_phase(cleanup)
    assert qualified.allowed_host.result.reachable is True
    attestation = qualified.provider_attestation
    assert attestation is not None
    assert attestation.network_egress_allowlist_enforced is True


def test_non_allowlisted_host_is_blocked(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    qualified = qualification_runner.run_qualified_phase(cleanup)
    assert qualified.denied_host.result.reachable is False


def test_redirect_escape_is_blocked(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    redirect = qualification_runner.run_redirect_phase(cleanup)
    assert redirect.redirect.attempted is True
    assert redirect.redirect.escaped is False


def test_attestation_correlation_passes(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    qualified = qualification_runner.run_qualified_phase(cleanup)
    correlation = qualified.attestation_correlation
    assert correlation is not None
    assert correlation.passes() is True
    assert correlation.attestation_verified is True


def test_physical_egress_causal_proof_end_to_end(
    qualification_runner: QualificationRunner,
) -> None:
    evidence = qualification_runner.run_and_assert()
    assert evidence.passes_causal_proof() is True
    _EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _EVIDENCE_ROOT / f"{evidence.execution_reference.replace(':', '-')}.json"
    serialized = json.dumps(evidence.to_mapping(), indent=2, sort_keys=True)
    for pattern in _SECRET_PATTERNS:
        assert pattern.search(serialized) is None
    path.write_text(serialized, encoding="utf-8")


def test_empty_allowlist_fails_closed(reference_backend: ReferenceSandboxBackend) -> None:
    with pytest.raises(ReferenceSubstrateSecurityError):
        reference_backend.create_session_with_security(
            SandboxSecurityRequirements(
                isolation_tier="cloud",
                network_egress="allowlist",
                network_egress_allowlist=canonicalize_network_egress_allowlist([]),
            ),
        )


def test_allowlist_does_not_authorize_denied_host(
    reference_backend: ReferenceSandboxBackend,
) -> None:
    scenario = default_reference_scenario()
    session = HostedSandboxSession.open(
        reference_backend,
        tenant_id="qual-tenant",
        task_id="qual-task",
        security_requirements=SandboxSecurityRequirements(
            isolation_tier="cloud",
            network_egress="allowlist",
            network_egress_allowlist=scenario.allowlist,
        ),
    )
    assert session is not None
    try:
        caps = session.security_capabilities()
        assert caps.network_egress_allowlist_enforced is True
        assert caps.enforced_network_hosts is not None
        enforced = {host.canonical_form() for host in caps.enforced_network_hosts.hosts}
        assert scenario.allowed_host in enforced
        assert scenario.denied_host not in enforced
    finally:
        reference_backend.destroy_session(session.session_id)


class _RecordingProbe:
    def __init__(self, *, reachable: bool) -> None:
        self._reachable = reachable

    def execute(self, session: HostedSandboxSession, target: str) -> NetworkProbeResult:
        return NetworkProbeResult(
            reachable=self._reachable,
            status_code=200 if self._reachable else None,
            redirect_target=None,
            latency_ms=0.0,
            redirected=False,
        )


def test_failure_during_probe_still_cleans_up(reference_backend: ReferenceSandboxBackend) -> None:
    provider = QualificationSandboxProvider(reference_backend)
    scenario = default_reference_scenario()
    runner = QualificationRunner(provider, scenario, probe=_RecordingProbe(reachable=False))
    cleanup: list = []
    control = runner.run_control_phase(cleanup)
    assert control.baseline_valid is False
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_policy_application_failure_does_not_leave_session(
    reference_backend: ReferenceSandboxBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_apply(*args: object, **kwargs: object) -> None:
        raise ReferenceSubstratePolicyError("simulated apply failure")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.apply_egress_policy_netns",
        _fail_apply,
    )
    scenario = default_reference_scenario()
    with pytest.raises(ReferenceSubstrateSecurityError):
        reference_backend.create_session_with_security(
            SandboxSecurityRequirements(
                isolation_tier="cloud",
                network_egress="allowlist",
                network_egress_allowlist=scenario.allowlist,
            ),
        )


def test_missing_kernel_evidence_cannot_assert_enforced(
    reference_backend: ReferenceSandboxBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = default_reference_scenario()
    session = reference_backend.create_session_with_security(
        SandboxSecurityRequirements(
            isolation_tier="cloud",
            network_egress="allowlist",
            network_egress_allowlist=scenario.allowlist,
        ),
    )

    def _fail_read(*args: object, **kwargs: object) -> None:
        raise ReferenceSubstratePolicyError("missing kernel evidence")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.read_verified_egress_policy",
        _fail_read,
    )
    with pytest.raises(ReferenceSubstrateSecurityError):
        reference_backend.session_security_capabilities(session.session_id)
    reference_backend.destroy_session(session.session_id)
