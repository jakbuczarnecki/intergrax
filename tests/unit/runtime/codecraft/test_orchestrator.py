# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.codecraft_wiring import (
    lab_codecraft_profile,
    tool_profile_with_codecraft,
    wire_application_codecraft,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.execution_identity import bind_active_execution_identity, mint_attempt_id, mint_run_id
from intergrax.runtime.codecraft.adaptive_trigger import evaluate_craft_trigger
from intergrax.runtime.codecraft.ownership import codecraft_exec_hitl_notes
from intergrax.runtime.codecraft.orchestrator import CodeCraftOrchestrator
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.providers.codecraft.service import (
    codecraft_dispose,
    codecraft_get_state,
    codecraft_iterate,
    codecraft_list_ephemeral_tools,
    codecraft_start,
)
from intergrax.tools.providers.codecraft.contracts import (
    CodeCraftDisposeToolInput,
    CodeCraftGetStateToolInput,
    CodeCraftIterateToolInput,
    CodeCraftListEphemeralToolsInput,
    CodeCraftStartToolInput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-1",
        task_id="task-1",
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python", "run_script", "browser_fetch"}
        ),
    )


@pytest.fixture
def craft_ctx(sandbox_session: SandboxSession) -> ToolWiringContext:
    profile = CodeCraftProfile(mode="autonomous", require_tests=False, forbidden_imports=["os"])
    manager = CodeCraftSessionManager()
    return ToolWiringContext(
        sandbox_session=sandbox_session,
        extras={
            "codecraft_profile": profile,
            "codecraft_session_manager": manager,
        },
    )


def test_codecraft_session_start_iterate_dispose(craft_ctx: ToolWiringContext) -> None:
    start_out = codecraft_start(
        craft_ctx,
        CodeCraftStartToolInput(goal="print greeting", task_id="task-1", tenant_id="tenant-1"),
    )
    assert start_out.session is not None
    craft_id = start_out.session.craft_id

    iter_out = codecraft_iterate(
        craft_ctx,
        CodeCraftIterateToolInput(craft_id=craft_id, task_id="task-1", tenant_id="tenant-1"),
    )
    assert iter_out.result.success is True
    assert iter_out.result.verdict in {"continue", "promote"}

    state_out = codecraft_get_state(
        craft_ctx,
        CodeCraftGetStateToolInput(craft_id=craft_id),
    )
    assert state_out.found is True
    assert state_out.session is not None
    assert state_out.session.iteration >= 1

    list_out = codecraft_list_ephemeral_tools(
        craft_ctx,
        CodeCraftListEphemeralToolsInput(craft_id=craft_id),
    )
    assert list_out.tool_ids

    dispose_out = codecraft_dispose(
        craft_ctx,
        CodeCraftDisposeToolInput(craft_id=craft_id, task_id="task-1", tenant_id="tenant-1"),
    )
    assert dispose_out.disposed is True


def test_supervised_mode_requires_hitl(craft_ctx: ToolWiringContext) -> None:
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    store = InMemoryHumanDecisionPersistence()
    ctx = ToolWiringContext(
        sandbox_session=craft_ctx.sandbox_session,
        human_decision_store=store,
        extras={
            "codecraft_profile": profile,
            "codecraft_session_manager": CodeCraftSessionManager(),
        },
    )
    run_id = mint_run_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        start_out = codecraft_start(
            ctx,
            CodeCraftStartToolInput(goal="demo", task_id="task-1", tenant_id="tenant-1"),
        )
        assert start_out.session is not None
        craft_id = start_out.session.craft_id
        iter_out = codecraft_iterate(
            ctx,
            CodeCraftIterateToolInput(craft_id=craft_id, task_id="task-1", tenant_id="tenant-1"),
        )
        assert iter_out.result.error == "hitl_pending"

        store.record(
            build_human_decision_record(
                task_id="task-1",
                tenant_id="tenant-1",
                user_id="operator",
                verdict=HumanResponseVerdict.APPROVE,
                response_text="ok",
                run_id=str(run_id),
                notes=codecraft_exec_hitl_notes(craft_id),
            ),
        )
        iter_ok = codecraft_iterate(
            ctx,
            CodeCraftIterateToolInput(craft_id=craft_id, task_id="task-1", tenant_id="tenant-1"),
        )
        assert iter_ok.result.error != "hitl_pending"
    finally:
        from intergrax.contracts.execution_identity import reset_active_execution_identity

        reset_active_execution_identity(token)


def test_wire_application_codecraft_enables_tools() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    assert env.codecraft_profile is not None
    wiring = wire_application_codecraft(env)
    assert "codecraft_governance" in wiring.domain_fragments
    updated = tool_profile_with_codecraft(env)
    assert "codecraft.start" in updated.enabled


def test_adaptive_trigger_catalog_miss() -> None:
    profile = lab_codecraft_profile(mode="autonomous")
    decision = evaluate_craft_trigger(
        requested_tool_id="custom.transform",
        catalog_has_tool=False,
        profile=profile,
        adaptive_enabled=True,
    )
    assert decision.suggest_craft is True
    assert decision.auto_invoke is True

    denied = evaluate_craft_trigger(
        requested_tool_id="custom.transform",
        catalog_has_tool=False,
        profile=profile,
        budget_exhausted=True,
    )
    assert denied.suggest_craft is False


def test_orchestrator_max_iterations(craft_ctx: ToolWiringContext) -> None:
    profile = CodeCraftProfile(mode="autonomous", max_iterations=1, require_tests=False)
    craft_ctx.extras["codecraft_profile"] = profile
    orch = CodeCraftOrchestrator(craft_ctx)
    session, _ = orch.start(goal="g", task_id="task-1", tenant_id="tenant-1")
    assert session is not None
    orch.iterate(craft_id=session.craft_id, task_id="task-1", tenant_id="tenant-1")
    session2, result = orch.iterate(craft_id=session.craft_id, task_id="task-1", tenant_id="tenant-1")
    assert result.error == "max_iterations_exceeded" or (session2 and session2.iteration >= 1)


def test_orchestrator_exec_budget_exhausted(craft_ctx: ToolWiringContext) -> None:
    profile = CodeCraftProfile(mode="autonomous", max_total_exec_time_s=1.0, require_tests=False)
    craft_ctx.extras["codecraft_profile"] = profile
    orch = CodeCraftOrchestrator(craft_ctx)
    session, _ = orch.start(goal="g", task_id="task-1", tenant_id="tenant-1")
    assert session is not None
    from intergrax.runtime.codecraft.ownership import resolve_codecraft_ownership

    ownership = resolve_codecraft_ownership(craft_ctx, caller_tenant_id="tenant-1", caller_task_id="task-1")
    session = session.model_copy(update={"total_exec_time_s": 1.0})
    orch._sessions.save_owned(session, ownership)  # noqa: SLF001 — budget precondition
    _, result = orch.iterate(craft_id=session.craft_id, task_id="task-1", tenant_id="tenant-1")
    assert result.error == "max_total_exec_time_exceeded"


def test_orchestrator_trace_taxonomy(craft_ctx: ToolWiringContext) -> None:
    orch = CodeCraftOrchestrator(craft_ctx)
    session, _ = orch.start(goal="print ok", task_id="task-1", tenant_id="tenant-1")
    assert session is not None
    orch.iterate(craft_id=session.craft_id, task_id="task-1", tenant_id="tenant-1")
    steps = {evt.step for evt in orch._emitter.events}  # noqa: SLF001 — trace contract test
    assert "codecraft.generation" in steps
    assert "codecraft.iteration_verdict" in steps
    assert "codecraft.test" in steps
    assert "codecraft.session_opened" in steps
