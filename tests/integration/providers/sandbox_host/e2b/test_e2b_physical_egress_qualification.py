# © Artur Czarnecki. All rights reserved.

"""E2B physical egress qualification — causal-proof integration tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.integrations.contracts.sandbox_host import SandboxExecResult, SandboxSession
from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from tests.integration.providers.sandbox_host.e2b.qualification import (
    HostProbeEvidence,
    NetworkProbeResult,
    PhysicalEgressQualificationEvidence,
    QualificationBaselineError,
    QualificationRunner,
    QualificationSandboxProvider,
    QualifiedPhaseEvidence,
    RedirectPhaseEvidence,
    default_e2b_physical_egress_scenario,
)
from tests.integration.providers.sandbox_host.e2b.qualification.models import RedirectEvidence

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.sandbox_provider,
    pytest.mark.qualification,
]

_QUALIFICATION_ROOT = Path(__file__).resolve().parent / "qualification"


def _credential_available() -> bool:
    try:
        E2bSandboxHostConfig.from_env().resolved_api_key()
    except Exception:
        return False
    return True


def _require_credentials() -> None:
    if not _credential_available():
        pytest.skip("E2B_API_KEY / INTERGRAX_E2B_API_KEY unavailable for physical qualification")


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


def test_control_phase_requires_baseline_connectivity(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    control = qualification_runner.run_control_phase(cleanup)
    assert control.baseline_valid is True
    assert control.allowed_host.result.reachable is True
    assert control.denied_host.result.reachable is True
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_qualified_phase_allows_declared_host(qualification_runner: QualificationRunner) -> None:
    cleanup: list = []
    qualified = qualification_runner.run_qualified_phase(cleanup)
    assert qualified.allowed_host.result.reachable is True
    attestation = qualified.provider_attestation
    assert attestation is not None
    assert attestation.network_egress_allowlist_enforced is True
    assert len(cleanup) == 1
    assert cleanup[0].destroyed is True


def test_qualified_phase_blocks_undeclared_host(qualification_runner: QualificationRunner) -> None:
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


class _RecordingProbe:
    def __init__(self, *, reachable: bool = True) -> None:
        self._reachable = reachable

    def execute(self, session: HostedSandboxSession, target: str) -> NetworkProbeResult:
        return NetworkProbeResult(
            reachable=self._reachable,
            status_code=200 if self._reachable else None,
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


def test_cleanup_runs_after_success() -> None:
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


def test_cleanup_runs_after_failure() -> None:
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
        provider=scenario.provider,
        control_phase=control,
        qualified_phase=QualifiedPhaseEvidence(
            allowed_host=HostProbeEvidence(scenario.allowed_host, NetworkProbeResult(True, 200, False)),
            denied_host=HostProbeEvidence(scenario.denied_host, NetworkProbeResult(False, None, False)),
            provider_attestation=None,
        ),
        redirect_phase=RedirectPhaseEvidence(
            redirect=RedirectEvidence(
                attempted=True,
                escaped=False,
                redirect_url=scenario.redirect_url,
                result=NetworkProbeResult(False, None, False),
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


def test_missing_credentials_skip_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        f"{__name__}._credential_available",
        lambda: False,
    )
    with pytest.raises(pytest.skip.Exception):  # type: ignore[attr-defined]
        _require_credentials()


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
