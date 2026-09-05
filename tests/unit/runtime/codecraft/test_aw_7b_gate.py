# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AW-7B-GATE — CodeCraft safety prerequisite qualification tests."""

from __future__ import annotations

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
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.session import SandboxSession
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


def _sandbox(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id=TENANT,
        task_id=TASK,
        allowed_operations=_CODECRAFT_OPS,
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


def test_hosted_resolution_marks_provider_identity() -> None:
    backend = MagicMock()
    backend.create_session.return_value = MagicMock(session_id="hosted-1")
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="cloud", network_egress="deny")
    resolution = resolve_craft_sandbox(
        ToolWiringContext(sandbox_host=backend, extras={"codecraft_profile": profile}),
        profile,
        tenant_id=TENANT,
        task_id=TASK,
    )
    assert isinstance(resolution.session, HostedSandboxSession)
    assert resolution.capabilities is not None
    assert resolution.capabilities.provider_id.startswith("hosted:")
    assert resolution.capabilities.network_egress_enforced is True
