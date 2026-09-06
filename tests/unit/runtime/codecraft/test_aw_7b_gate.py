# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AW-7B-GATE — CodeCraft safety prerequisite qualification tests."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.codecraft.promoter import CraftResultPromoter
from intergrax.codecraft.static_gate import StaticCodeGate
from intergrax.contracts.execution_identity import bind_active_execution_identity, mint_attempt_id, mint_run_id
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.orchestrator import CodeCraftOrchestrator
from intergrax.runtime.codecraft.ownership import codecraft_exec_hitl_notes
from intergrax.runtime.codecraft.sandbox_resolver import resolve_craft_sandbox_with_evidence
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.codecraft.substrate import resolve_craft_sandbox
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.integrations.contracts.sandbox_host import (
    SandboxArtifact,
    SandboxExecResult,
    SandboxHostBackend,
    SandboxSession as HostSandboxSession,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities, SandboxSecurityCapable
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.tools.providers.codecraft.contracts import (
    CodeCraftIterateToolInput,
    CodeCraftListEphemeralToolsInput,
    CodeCraftPromoteToolInput,
    CodeCraftRunToolInput,
    CodeCraftStartToolInput,
)
from intergrax.tools.providers.codecraft.service import (
    codecraft_iterate,
    codecraft_list_ephemeral_tools,
    codecraft_promote,
    codecraft_run,
    codecraft_start,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TENANT = "tenant-a"
TASK = "task-a"
RUN_ID = mint_run_id()
ATTEMPT_ID = mint_attempt_id()

_CODECRAFT_OPS = frozenset(
    {"echo", "write_file", "read_file", "list_files", "run_python", "run_script"},
)


class _PlainSandboxHostBackend:
    """Canonical ``SandboxHostBackend`` without ``SandboxSecurityCapable`` attestation."""

    def __init__(self, *, session_id: str = "hosted-plain") -> None:
        self._session_id = session_id

    def create_session(self) -> HostSandboxSession:
        return HostSandboxSession(session_id=self._session_id)

    def exec(self, session_id: str, command: str) -> SandboxExecResult:
        return SandboxExecResult()

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        return SandboxArtifact(artifact_id="artifact-1")


class _FakeHostedSecurityBackend:
    """Provider-neutral hosted backend that attests sandbox security capabilities."""

    def __init__(
        self,
        *,
        session_id: str = "hosted-1",
        provider_id: str = "fake-hosted",
        isolation_tier: str = "cloud",
        network_egress_deny_enforced: bool | None = None,
    ) -> None:
        self._session_id = session_id
        self._provider_id = provider_id
        self._isolation_tier = isolation_tier
        self._network_egress_deny_enforced = network_egress_deny_enforced

    def create_session(self):
        return MagicMock(session_id=self._session_id)

    def exec(self, session_id: str, command: str):
        return MagicMock(exit_code=0, stdout="", stderr="")

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        return SandboxArtifact(artifact_id="artifact-1")

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        return SandboxSecurityCapabilities(
            isolation_tier=self._isolation_tier,  # type: ignore[arg-type]
            provider_id=self._provider_id,
            network_egress_deny_enforced=self._network_egress_deny_enforced,
        )


def _sandbox(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id=TENANT,
        task_id=TASK,
        allowed_operations=_CODECRAFT_OPS,
    )


def _sandbox_env_profile() -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="codecraft-test")
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=True),
            ),
        },
    )


def _ctx(
    sandbox: SandboxSession,
    *,
    profile: CodeCraftProfile | None = None,
    manager: CodeCraftSessionManager | None = None,
    hitl_store: InMemoryHumanDecisionPersistence | None = None,
    registry: EphemeralToolRegistryStore | None = None,
) -> ToolWiringContext:
    extras: dict[str, object] = {
        "codecraft_session_manager": manager or CodeCraftSessionManager(),
        "codecraft_ephemeral_registry": registry or EphemeralToolRegistryStore(),
        "effective_environment_profile": _sandbox_env_profile(),
    }
    if profile is not None:
        extras["codecraft_profile"] = profile
    return ToolWiringContext(
        sandbox_session=sandbox,
        human_decision_store=hitl_store,
        extras=extras,
    )


def test_architecture_single_codecraft_orchestrator() -> None:
    assert CodeCraftOrchestrator.__name__ == "CodeCraftOrchestrator"


def test_local_isolation_resolves_with_egress_evidence(tmp_path: Path) -> None:
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="local")
    ctx = _ctx(_sandbox(tmp_path), profile=profile)
    resolution = resolve_craft_sandbox_with_evidence(ctx, profile, tenant_id=TENANT, task_id=TASK)
    assert resolution.session is not None
    assert resolution.capabilities is not None
    assert resolution.capabilities.resolved_tier == "local"
    assert resolution.capabilities.downgraded is False
    assert resolution.capabilities.network_egress_enforced is True


def test_container_without_hosted_fails_closed(tmp_path: Path) -> None:
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="container")
    ctx = _ctx(_sandbox(tmp_path), profile=profile)
    resolution = resolve_craft_sandbox(ctx, profile, tenant_id=TENANT, task_id=TASK)
    assert resolution.session is None
    assert resolution.error == "isolation_requirement_unsatisfied"


def test_cloud_without_hosted_fails_closed(tmp_path: Path) -> None:
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud")
    ctx = _ctx(_sandbox(tmp_path), profile=profile)
    resolution = resolve_craft_sandbox(ctx, profile, tenant_id=TENANT, task_id=TASK)
    assert resolution.session is None
    assert resolution.error == "isolation_requirement_unsatisfied"


def test_cloud_run_does_not_downgrade_to_local(tmp_path: Path) -> None:
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud")
    ctx = _ctx(_sandbox(tmp_path), profile=profile)
    with patch("intergrax.tools.providers.codecraft.service.code_exec") as mocked_exec:
        out = codecraft_run(
            ctx,
            CodeCraftRunToolInput(code="print('x')\n", tenant_id=TENANT, task_id=TASK),
        )
        mocked_exec.assert_not_called()
    assert out.result.error == "isolation_requirement_unsatisfied"


def test_egress_deny_fails_when_browser_fetch_allowed(tmp_path: Path) -> None:
    sandbox = SandboxSession.create(
        tmp_path,
        tenant_id=TENANT,
        task_id=TASK,
        allowed_operations=frozenset({*_CODECRAFT_OPS, "browser_fetch"}),
    )
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="local", network_egress="deny")
    resolution = resolve_craft_sandbox(_ctx(sandbox, profile=profile), profile, tenant_id=TENANT, task_id=TASK)
    assert resolution.session is None
    assert resolution.error == "network_egress_requirement_unsatisfied"


def test_static_gate_blocks_before_sandbox_exec(tmp_path: Path) -> None:
    profile = CodeCraftProfile(mode="autonomous")
    ctx = _ctx(_sandbox(tmp_path), profile=profile)
    with patch("intergrax.tools.providers.codecraft.service.code_exec") as mocked_exec:
        out = codecraft_run(
            ctx,
            CodeCraftRunToolInput(code="import os\nprint(1)\n", tenant_id=TENANT, task_id=TASK),
        )
        mocked_exec.assert_not_called()
    assert out.result.success is False
    assert "forbidden_import" in out.result.static_gate.rule_ids


def test_static_gate_order_architecture() -> None:
    profile = CodeCraftProfile(mode="autonomous")
    gate = StaticCodeGate(profile).scan("import os\n", language="python")
    assert gate.passed is False


def test_promotion_fail_closed_without_verification_evidence(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    ctx = _ctx(_sandbox(tmp_path), profile=profile, manager=manager)
    start = codecraft_start(
        ctx,
        CodeCraftStartToolInput(goal="demo", tenant_id=TENANT, task_id=TASK, initial_code="print('x')\n"),
    )
    assert start.session is not None
    out = codecraft_promote(
        ctx,
        CodeCraftPromoteToolInput(craft_id=start.session.craft_id, tenant_id=TENANT, task_id=TASK),
    )
    assert out.result.success is False
    assert out.result.error == "promotion_verification_missing"


def test_promotion_requires_cvl_promote_verdict(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False, max_iterations=8)
    ctx = _ctx(_sandbox(tmp_path), profile=profile, manager=manager)
    start = codecraft_start(
        ctx,
        CodeCraftStartToolInput(goal="demo", tenant_id=TENANT, task_id=TASK, initial_code="print('x')\n"),
    )
    assert start.session is not None
    iterate = codecraft_iterate(
        ctx,
        CodeCraftIterateToolInput(craft_id=start.session.craft_id, tenant_id=TENANT, task_id=TASK),
    )
    assert iterate.result.verdict in {"continue", "promote"}
    if iterate.result.verdict != "promote":
        promoter = CraftResultPromoter()
        session = iterate.session
        assert session is not None
        denied = promoter.promote_session(session)
        assert denied.success is False
        assert denied.error == "cvl_verdict_not_promote"


def test_ephemeral_registry_isolated_per_craft() -> None:
    store = EphemeralToolRegistryStore()
    store.for_craft("craft-a").register("ephemeral.craft-a.helper")
    store.for_craft("craft-b").register("ephemeral.craft-b.helper")
    assert store.for_craft("craft-a").list_tools() == ("ephemeral.craft-a.helper",)
    assert store.for_craft("craft-b").list_tools() == ("ephemeral.craft-b.helper",)


def test_ephemeral_tools_not_listed_for_foreign_craft(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    ctx = _ctx(_sandbox(tmp_path), profile=profile, manager=manager, registry=registry)
    start = codecraft_start(
        ctx,
        CodeCraftStartToolInput(
            goal="demo",
            tenant_id=TENANT,
            task_id=TASK,
            craft_id="craft-owned",
            initial_code="print('x')\n",
        ),
    )
    assert start.session is not None
    foreign = codecraft_list_ephemeral_tools(
        ctx,
        CodeCraftListEphemeralToolsInput(craft_id="craft-foreign", tenant_id=TENANT, task_id=TASK),
    )
    assert foreign.tool_ids == []


def test_hitl_approval_from_other_craft_rejected(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    ctx = _ctx(_sandbox(tmp_path), profile=profile, manager=CodeCraftSessionManager(), hitl_store=store)
    token = bind_active_execution_identity(run_id=RUN_ID, attempt_id=ATTEMPT_ID)
    try:
        start = codecraft_start(
            ctx,
            CodeCraftStartToolInput(goal="demo", tenant_id=TENANT, task_id=TASK, initial_code="print('x')\n"),
        )
        assert start.session is not None
        craft_id = start.session.craft_id
        store.record(
            build_human_decision_record(
                task_id=TASK,
                tenant_id=TENANT,
                approver=local_development_approver_evidence(tenant_id=TENANT, actor_id="operator"),
                verdict=HumanResponseVerdict.APPROVE,
                response_text="ok",
                run_id=str(RUN_ID),
                notes=codecraft_exec_hitl_notes("craft-other"),
            ),
        )
        with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
            out = codecraft_iterate(
                ctx,
                CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=TENANT, task_id=TASK),
            )
            mocked_exec.assert_not_called()
        assert out.result.error == "hitl_pending"
    finally:
        from intergrax.contracts.execution_identity import reset_active_execution_identity

        reset_active_execution_identity(token)


def test_hosted_unknown_capability_fails_closed_on_deny() -> None:
    backend = _PlainSandboxHostBackend(session_id="hosted-unknown")
    assert isinstance(backend, SandboxHostBackend)
    assert not isinstance(backend, SandboxSecurityCapable)
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_requirement_unsatisfied"


def test_hosted_session_unknown_backend_returns_unknown_evidence() -> None:
    backend = _PlainSandboxHostBackend(session_id="hosted-plain")
    session = HostedSandboxSession.open(backend, tenant_id=TENANT, task_id=TASK)
    caps = session.security_capabilities()
    assert caps.isolation_tier == "cloud"
    assert caps.provider_id == f"hosted:{session.session_id}"
    assert caps.network_egress_deny_enforced is None


def test_hosted_session_consumes_structural_sandbox_security_capable() -> None:
    backend = _FakeHostedSecurityBackend(
        session_id="hosted-structural",
        provider_id="fake-hosted:structural",
        network_egress_deny_enforced=True,
    )
    assert isinstance(backend, SandboxSecurityCapable)
    session = HostedSandboxSession.open(backend, tenant_id=TENANT, task_id=TASK)
    caps = session.security_capabilities()
    assert caps.provider_id == "fake-hosted:structural"
    assert caps.network_egress_deny_enforced is True


def test_hosted_positive_capability_allows_deny() -> None:
    backend = _FakeHostedSecurityBackend(
        session_id="hosted-proven",
        provider_id="fake-hosted:proven",
        network_egress_deny_enforced=True,
    )
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert isinstance(resolution.session, HostedSandboxSession)
    assert resolution.capabilities is not None
    assert resolution.capabilities.network_egress_enforced is True
    assert resolution.capabilities.provider_id == "fake-hosted:proven"


def test_hosted_false_capability_fails_closed_on_deny() -> None:
    backend = _FakeHostedSecurityBackend(network_egress_deny_enforced=False)
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert resolution.session is None
    assert resolution.error == "network_egress_requirement_unsatisfied"


def test_hosted_type_alone_does_not_prove_egress_deny() -> None:
    backend = MagicMock()
    backend.create_session.return_value = MagicMock(session_id="hosted-type-only")
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert isinstance(resolution.session, HostedSandboxSession) is False
    assert resolution.error == "network_egress_requirement_unsatisfied"


def test_local_capability_evidence_uses_public_contract(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    caps = sandbox.security_capabilities()
    assert caps.isolation_tier == "local"
    assert caps.network_egress_deny_enforced is True
    substrate_source = Path("intergrax/runtime/codecraft/substrate.py").read_text(encoding="utf-8")
    assert "_allowed_operations" not in substrate_source


def test_architecture_gate_blocks_private_sandbox_field_access() -> None:
    substrate_source = Path("intergrax/runtime/codecraft/substrate.py").read_text(encoding="utf-8")
    assert "_allowed_operations" not in substrate_source
    assert "_private_config" not in substrate_source
    assert "_backend" not in substrate_source


def test_architecture_gate_blocks_reflective_security_capability_discovery() -> None:
    """Security evidence must use ``SandboxSecurityCapable``, not getattr duck-typing."""
    source_path = Path("intergrax/runtime/sandbox/hosted_session.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_attr_names = {"security_capabilities", "provider_id"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            continue
        if len(node.args) < 2:
            continue
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and attr.value in forbidden_attr_names:
            pytest.fail(f"forbidden reflective getattr on {attr.value!r} in hosted_session security boundary")


def test_hosted_resolution_marks_provider_identity() -> None:
    backend = _FakeHostedSecurityBackend(
        session_id="hosted-1",
        provider_id="fake-hosted:identity",
        network_egress_deny_enforced=True,
    )
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert isinstance(resolution.session, HostedSandboxSession)
    assert resolution.capabilities is not None
    assert resolution.capabilities.provider_id == "fake-hosted:identity"
    assert resolution.capabilities.network_egress_enforced is True


@pytest.mark.parametrize("tier", ["container", "cloud"])
def test_strong_isolation_does_not_call_local_resolver(tmp_path: Path, tier: str) -> None:
    """Architecture gate: container/cloud must not fall back to resolve_sandbox_session."""
    profile = CodeCraftProfile(mode="autonomous", isolation_tier=tier)  # type: ignore[arg-type]
    ctx = _ctx(_sandbox(tmp_path), profile=profile)
    with patch("intergrax.runtime.codecraft.substrate.resolve_sandbox_session") as local_resolver:
        resolution = resolve_craft_sandbox(ctx, profile, tenant_id=TENANT, task_id=TASK)
        local_resolver.assert_not_called()
    assert resolution.session is None
    assert resolution.error == "isolation_requirement_unsatisfied"


def test_container_hosted_resolution_selects_hosted_session() -> None:
    backend = _FakeHostedSecurityBackend(
        session_id="hosted-container-1",
        isolation_tier="container",
        network_egress_deny_enforced=True,
    )
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="container", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert isinstance(resolution.session, HostedSandboxSession)
    assert resolution.capabilities is not None
    assert resolution.capabilities.resolved_tier == "container"
    assert resolution.capabilities.downgraded is False


def test_orchestrator_reuses_resolved_sandbox_for_tests(tmp_path: Path) -> None:
    """Execution and verification must share the same resolved sandbox context."""
    profile = CodeCraftProfile(mode="autonomous", require_tests=True, max_iterations=2)
    ctx = _ctx(_sandbox(tmp_path), profile=profile)
    orchestrator = CodeCraftOrchestrator(ctx)
    start_session, _ = orchestrator.start(
        goal="demo",
        tenant_id=TENANT,
        task_id=TASK,
        initial_code="print('ok')\n",
    )
    assert start_session is not None
    captured_sessions: list[object] = []

    def _capture_run(self, wiring_ctx, *, rel_path="craft_main.py", sandbox_session=None):
        captured_sessions.append(sandbox_session)
        from intergrax.codecraft.test_runner import CraftTestResult

        return CraftTestResult(
            passed=True,
            skipped=False,
            command="pytest craft_main.py",
            stdout="",
            stderr="",
            exit_code=0,
        )

    with patch("intergrax.runtime.codecraft.orchestrator.CraftTestRunner.run", _capture_run):
        with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
            mocked_exec.return_value = MagicMock(
                success=True,
                session_id="exec-sandbox-1",
                output={"stdout": "ok\n", "stderr": "", "exit_code": 0},
                error="",
            )
            session, result = orchestrator.iterate(
                craft_id=start_session.craft_id,
                tenant_id=TENANT,
                task_id=TASK,
            )
    assert session is not None
    assert result.success is True
    assert captured_sessions, "CraftTestRunner must receive resolved sandbox"
    assert captured_sessions[0] is not None
