# © Artur Czarnecki. All rights reserved.

"""E2B physical egress qualification — causal-proof integration tests."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from intergrax.integrations.contracts.sandbox_host import SandboxExecResult, SandboxSession
from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from tests.integration.providers.sandbox_host.e2b.qualification import (
    E2bCredentialStatus,
    HostProbeEvidence,
    NetworkProbeResult,
    PhysicalEgressQualificationEvidence,
    QualificationBaselineError,
    QualificationRunner,
    QualificationSandboxProvider,
    QualifiedPhaseEvidence,
    RedirectPhaseEvidence,
    default_e2b_physical_egress_scenario,
    resolve_e2b_credentials,
)
from tests.integration.providers.sandbox_host.e2b.qualification.models import RedirectEvidence

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.sandbox_provider,
    pytest.mark.qualification,
]

_QUALIFICATION_ROOT = Path(__file__).resolve().parent / "qualification"
_EVIDENCE_ROOT = Path(".tmp/session/e2b-physical-egress-qualification")
_CREDENTIAL_SKIP_REASON = "E2B credentials unavailable from environment"
_SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9]{8,}"),
    re.compile(r"E2B_API_KEY", re.IGNORECASE),
    re.compile(r"INTERGRAX_E2B_API_KEY", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}"),
)


def _require_credentials() -> None:
    if resolve_e2b_credentials() is not E2bCredentialStatus.AVAILABLE:
        pytest.skip(_CREDENTIAL_SKIP_REASON)


def _write_qualification_evidence(evidence: PhysicalEgressQualificationEvidence) -> Path:
    _EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    safe_ref = evidence.execution_reference.replace(":", "-")
    evidence_path = _EVIDENCE_ROOT / f"{safe_ref}.json"
    payload = evidence.to_mapping()
    payload["session_identifier_hashes"] = [
        hashlib.sha256(cleanup.session_id.encode("utf-8")).hexdigest()[:16]
        for cleanup in evidence.cleanup_phases
    ]
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    for pattern in _SECRET_PATTERNS:
        assert pattern.search(serialized) is None
    evidence_path.write_text(serialized, encoding="utf-8")
    return evidence_path


@pytest.fixture(scope="module")
def qualification_backend():
    _require_credentials()
    try:
        backend = create_e2b_sandbox_host().client
    except Exception as exc:
        pytest.skip(f"E2B backend unavailable: {type(exc).__name__}")
    assert backend is not None
    yield backend


@pytest.fixture(scope="module")
def qualification_runner(qualification_backend) -> QualificationRunner:
    provider = QualificationSandboxProvider(qualification_backend)
    return QualificationRunner(provider, default_e2b_physical_egress_scenario())


def test_control_environment_proves_baseline_connectivity(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    control = qualification_runner.run_control_phase(cleanup)
    assert control.baseline_valid is True
    assert control.allowed_host.result.reachable is True
    assert control.denied_host.result.reachable is True
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_allowlisted_host_is_reachable(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    qualified = qualification_runner.run_qualified_phase(cleanup)
    assert qualified.allowed_host.result.reachable is True
    attestation = qualified.provider_attestation
    assert attestation is not None
    assert attestation.network_egress_allowlist_enforced is True
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_non_allowlisted_host_is_blocked(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    qualified = qualification_runner.run_qualified_phase(cleanup)
    assert qualified.denied_host.result.reachable is False
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_redirect_escape_is_blocked(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    redirect = qualification_runner.run_redirect_phase(cleanup)
    evidence = redirect.redirect
    assert evidence.attempted is True
    assert evidence.escaped is False
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_attestation_correlation_passes(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    qualified = qualification_runner.run_qualified_phase(cleanup)
    correlation = qualified.attestation_correlation
    assert correlation is not None
    assert correlation.passes() is True
    assert correlation.attestation_verified is True
    assert correlation.execution_verified is True
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_physical_egress_causal_proof_end_to_end(
    qualification_runner: QualificationRunner,
) -> None:
    evidence = qualification_runner.run_and_assert()
    assert evidence.passes_causal_proof() is True
    evidence_path = _write_qualification_evidence(evidence)
    assert evidence_path.is_file()


class _RecordingProbe:
    def __init__(self, *, reachable: bool = True) -> None:
        self._reachable = reachable

    def execute(self, session: HostedSandboxSession, target: str) -> NetworkProbeResult:
        return NetworkProbeResult(
            reachable=self._reachable,
            status_code=200 if self._reachable else None,
            redirect_target=None,
            latency_ms=0.0,
            redirected=False,
        )


class _FakeQualificationBackend:
    def __init__(self, *, fail_destroy: bool = False) -> None:
        self.fail_destroy = fail_destroy
        self.destroy_calls: list[str] = []
        self._counter = 0

    def create_session(self) -> SandboxSession:
        self._counter += 1
        return SandboxSession(session_id=f"fake-{self._counter}", status="running")

    def create_session_with_security(self, requirements):  # noqa: ANN001 — fake boundary
        return self.create_session()

    def destroy_session(self, session_id: str) -> None:
        if self.fail_destroy:
            raise RuntimeError("destroy failed")
        self.destroy_calls.append(session_id)

    def exec(self, session_id: str, command: str) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr="", exit_code=0)


def test_cleanup_after_success() -> None:
    backend = _FakeQualificationBackend()
    provider = QualificationSandboxProvider(backend)
    runner = QualificationRunner(
        provider,
        default_e2b_physical_egress_scenario(),
        probe=_RecordingProbe(reachable=True),
    )
    evidence = runner.run_control_phase([])
    assert evidence.baseline_valid is True
    assert backend.destroy_calls == ["fake-1"]


def test_cleanup_after_failure() -> None:
    backend = _FakeQualificationBackend()
    provider = QualificationSandboxProvider(backend)
    scenario = default_e2b_physical_egress_scenario()
    runner = QualificationRunner(
        provider,
        scenario,
        probe=_RecordingProbe(reachable=False),
    )
    cleanup: list = []
    control = runner.run_control_phase(cleanup)
    assert control.baseline_valid is False
    assert backend.destroy_calls == ["fake-1"]
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True
    failing_evidence = PhysicalEgressQualificationEvidence(
        scenario_id=scenario.scenario_id,
        provider_identity=scenario.provider,
        execution_reference="test-execution-ref",
        timestamp_utc="2026-09-08T00:00:00+00:00",
        control_phase=control,
        qualified_phase=QualifiedPhaseEvidence(
            allowed_host=HostProbeEvidence(
                scenario.allowed_host,
                NetworkProbeResult(True, 200, None, 0.0, False),
            ),
            denied_host=HostProbeEvidence(
                scenario.denied_host,
                NetworkProbeResult(False, None, None, 0.0, False),
            ),
            provider_attestation=None,
        ),
        redirect_phase=RedirectPhaseEvidence(
            redirect=RedirectEvidence(
                attempted=True,
                escaped=False,
                redirect_url=scenario.redirect_url,
                result=NetworkProbeResult(False, None, None, 0.0, False),
            ),
        ),
        cleanup_phases=tuple(cleanup),
    )
    with pytest.raises(QualificationBaselineError):
        QualificationRunner.assert_causal_proof(failing_evidence)


def test_cleanup_runs_after_failure_records_cleanup_error() -> None:
    backend = _FakeQualificationBackend(fail_destroy=True)
    provider = QualificationSandboxProvider(backend)
    runner = QualificationRunner(
        provider,
        default_e2b_physical_egress_scenario(),
        probe=_RecordingProbe(reachable=True),
    )
    cleanup: list = []
    runner.run_control_phase(cleanup)
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is False
    assert cleanup[0].error is not None


def test_missing_credentials_skip_without_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.integration.providers.sandbox_host.e2b.qualification.credentials.resolve_e2b_credentials",
        lambda: E2bCredentialStatus.UNAVAILABLE,
    )
    with pytest.raises(pytest.skip.Exception) as exc_info:  # type: ignore[attr-defined]
        _require_credentials()
    assert _CREDENTIAL_SKIP_REASON in str(exc_info.value)


_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.nexus",
    "intergrax.runtime.nexus",
    "intergrax.autonomous_work",
    "applications.",
    "intergrax.applications",
)


def _collect_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


@pytest.mark.parametrize(
    "path",
    sorted(_QUALIFICATION_ROOT.rglob("*.py")),
    ids=lambda path: path.relative_to(_QUALIFICATION_ROOT).as_posix(),
)
def test_no_nexus_dependency(path: Path) -> None:
    joined = "\n".join(_collect_imports(path.read_text(encoding="utf-8"))).lower()
    for prefix in _FORBIDDEN_IMPORT_PREFIXES:
        assert prefix.lower() not in joined, f"{path} imports forbidden surface {prefix}"
