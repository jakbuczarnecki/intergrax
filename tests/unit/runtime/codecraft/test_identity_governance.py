# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CODECRAFT-01/02 identity governance negative tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.execution_identity import bind_active_execution_identity, mint_attempt_id, mint_run_id
from intergrax.runtime.codecraft.ownership import codecraft_exec_hitl_notes
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.providers.codecraft.contracts import (
    CodeCraftDisposeToolInput,
    CodeCraftGetStateToolInput,
    CodeCraftIterateToolInput,
    CodeCraftPromoteToolInput,
    CodeCraftRunToolInput,
    CodeCraftStartToolInput,
)
from intergrax.tools.providers.codecraft.service import (
    codecraft_dispose,
    codecraft_get_state,
    codecraft_iterate,
    codecraft_promote,
    codecraft_run,
    codecraft_start,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
TASK_X = "task-x"
TASK_Y = "task-y"
CRAFT_FIXED = "craft-fixed"
RUN_A = mint_run_id()
ATTEMPT_A = mint_attempt_id()


def _sandbox(tmp_path: Path, *, tenant_id: str, task_id: str) -> SandboxSession:
    return SandboxSession.create(
        tmp_path / tenant_id / task_id,
        tenant_id=tenant_id,
        task_id=task_id,
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python", "run_script", "browser_fetch"}
        ),
    )


def _ctx(
    sandbox: SandboxSession,
    *,
    profile: CodeCraftProfile | None = None,
    manager: CodeCraftSessionManager | None = None,
    hitl_store: InMemoryHumanDecisionPersistence | None = None,
) -> ToolWiringContext:
    extras: dict[str, object] = {
        "codecraft_session_manager": manager or CodeCraftSessionManager(),
    }
    if profile is not None:
        extras["codecraft_profile"] = profile
    return ToolWiringContext(
        sandbox_session=sandbox,
        human_decision_store=hitl_store,
        extras=extras,
    )


def _open_session(
    ctx: ToolWiringContext,
    *,
    tenant_id: str,
    task_id: str,
    craft_id: str = CRAFT_FIXED,
) -> str:
    out = codecraft_start(
        ctx,
        CodeCraftStartToolInput(
            goal="print ok",
            tenant_id=tenant_id,
            task_id=task_id,
            craft_id=craft_id,
            initial_code="print('ok')\n",
        ),
    )
    assert out.session is not None, out.error
    return out.session.craft_id


def _approve(
    store: InMemoryHumanDecisionPersistence,
    *,
    tenant_id: str,
    task_id: str,
    craft_id: str,
    run_id: str | None = None,
) -> None:
    store.record(
        build_human_decision_record(
            task_id=task_id,
            tenant_id=tenant_id,
            user_id="operator",
            verdict=HumanResponseVerdict.APPROVE,
            response_text="approved",
            run_id=run_id,
            notes=codecraft_exec_hitl_notes(craft_id),
        ),
    )


def test_cross_tenant_get_state(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    ctx_b = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_B, task_id=TASK_X),
        profile=CodeCraftProfile(mode="autonomous", require_tests=False),
        manager=manager,
    )
    craft_id = _open_session(ctx_b, tenant_id=TENANT_B, task_id=TASK_X)

    ctx_a = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X),
        profile=CodeCraftProfile(mode="autonomous", require_tests=False),
        manager=manager,
    )
    state = codecraft_get_state(
        ctx_a,
        CodeCraftGetStateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert state.found is False
    assert state.session is None

    owner_state = codecraft_get_state(
        ctx_b,
        CodeCraftGetStateToolInput(craft_id=craft_id, tenant_id=TENANT_B, task_id=TASK_X),
    )
    assert owner_state.found is True


def test_cross_task_iterate_unchanged(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    ctx_x = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile, manager=manager)
    craft_id = _open_session(ctx_x, tenant_id=TENANT_A, task_id=TASK_X)
    before = codecraft_get_state(
        ctx_x,
        CodeCraftGetStateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert before.session is not None
    snapshot = before.session.model_copy(deep=True)

    ctx_y = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_Y), profile=profile, manager=manager)
    with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
        out = codecraft_iterate(
            ctx_y,
            CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_Y),
        )
        mocked_exec.assert_not_called()

    assert out.result.error == "craft_session_ownership_mismatch"
    after = codecraft_get_state(
        ctx_x,
        CodeCraftGetStateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert after.session == snapshot


def test_cross_tenant_dispose(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    ctx_b = _ctx(_sandbox(tmp_path, tenant_id=TENANT_B, task_id=TASK_X), profile=profile, manager=manager)
    craft_id = _open_session(ctx_b, tenant_id=TENANT_B, task_id=TASK_X)

    ctx_a = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile, manager=manager)
    disposed = codecraft_dispose(
        ctx_a,
        CodeCraftDisposeToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert disposed.disposed is False

    owner_state = codecraft_get_state(
        ctx_b,
        CodeCraftGetStateToolInput(craft_id=craft_id, tenant_id=TENANT_B, task_id=TASK_X),
    )
    assert owner_state.found is True


def test_cross_tenant_promote(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    ctx_b = _ctx(_sandbox(tmp_path, tenant_id=TENANT_B, task_id=TASK_X), profile=profile, manager=manager)
    craft_id = _open_session(ctx_b, tenant_id=TENANT_B, task_id=TASK_X)

    ctx_a = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile, manager=manager)
    out = codecraft_promote(
        ctx_a,
        CodeCraftPromoteToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert out.result.error == "craft_session_ownership_mismatch"
    assert out.result.success is False


def test_craft_id_collision_same_scope(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    ctx = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile, manager=manager)
    first = codecraft_start(
        ctx,
        CodeCraftStartToolInput(
            goal="first",
            tenant_id=TENANT_A,
            task_id=TASK_X,
            craft_id=CRAFT_FIXED,
            initial_code="print('first')\n",
        ),
    )
    assert first.session is not None

    second = codecraft_start(
        ctx,
        CodeCraftStartToolInput(
            goal="second",
            tenant_id=TENANT_A,
            task_id=TASK_X,
            craft_id=CRAFT_FIXED,
            initial_code="print('second')\n",
        ),
    )
    assert second.error == "craft_session_already_open"

    state = codecraft_get_state(
        ctx,
        CodeCraftGetStateToolInput(craft_id=CRAFT_FIXED, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert state.session is not None
    assert state.session.goal == "first"


def test_craft_id_collision_different_tenant(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="autonomous", require_tests=False)
    ctx_a = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile, manager=manager)
    first = codecraft_start(
        ctx_a,
        CodeCraftStartToolInput(
            goal="tenant-a",
            tenant_id=TENANT_A,
            task_id=TASK_X,
            craft_id=CRAFT_FIXED,
            initial_code="print('a')\n",
        ),
    )
    assert first.session is not None

    ctx_b = _ctx(_sandbox(tmp_path, tenant_id=TENANT_B, task_id=TASK_X), profile=profile, manager=manager)
    second = codecraft_start(
        ctx_b,
        CodeCraftStartToolInput(
            goal="tenant-b",
            tenant_id=TENANT_B,
            task_id=TASK_X,
            craft_id=CRAFT_FIXED,
            initial_code="print('b')\n",
        ),
    )
    assert second.error == "craft_session_ownership_conflict"

    owner_state = codecraft_get_state(
        ctx_a,
        CodeCraftGetStateToolInput(craft_id=CRAFT_FIXED, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert owner_state.session is not None
    assert owner_state.session.goal == "tenant-a"


def test_iterate_input_rejects_hitl_approved_field() -> None:
    with pytest.raises(ValidationError):
        CodeCraftIterateToolInput(
            craft_id="craft-1",
            hitl_approved=True,  # type: ignore[call-arg]
        )


def test_caller_bool_cannot_authorize_iterate(tmp_path: Path) -> None:
    manager = CodeCraftSessionManager()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    ctx = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile, manager=manager)
    craft_id = _open_session(ctx, tenant_id=TENANT_A, task_id=TASK_X)

    with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
        out = codecraft_iterate(
            ctx,
            CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
        )
        mocked_exec.assert_not_called()

    assert out.result.error == "hitl_pending"


def test_hitl_required_without_decision_store(tmp_path: Path) -> None:
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True)
    ctx = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile)
    with patch("intergrax.tools.providers.codecraft.service.code_exec") as mocked_exec:
        out = codecraft_run(
            ctx,
            CodeCraftRunToolInput(
                code="print('nope')\n",
                tenant_id=TENANT_A,
                task_id=TASK_X,
            ),
        )
        mocked_exec.assert_not_called()
    assert out.result.error == "hitl_pending"
    assert out.result.success is False


def test_wrong_tenant_decision_does_not_authorize(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    ctx = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X),
        profile=profile,
        hitl_store=store,
    )
    craft_id = _open_session(ctx, tenant_id=TENANT_A, task_id=TASK_X)
    _approve(store, tenant_id=TENANT_B, task_id=TASK_X, craft_id=craft_id)

    with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
        out = codecraft_iterate(
            ctx,
            CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
        )
        mocked_exec.assert_not_called()
    assert out.result.error == "hitl_pending"


def test_rejected_decision_blocks_execution(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    ctx = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X),
        profile=profile,
        hitl_store=store,
    )
    craft_id = _open_session(ctx, tenant_id=TENANT_A, task_id=TASK_X)
    store.record(
        build_human_decision_record(
            task_id=TASK_X,
            tenant_id=TENANT_A,
            user_id="operator",
            verdict=HumanResponseVerdict.REJECT,
            response_text="no",
            notes=codecraft_exec_hitl_notes(craft_id),
        ),
    )

    with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
        out = codecraft_iterate(
            ctx,
            CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
        )
        mocked_exec.assert_not_called()
    assert out.result.error == "hitl_denied"


def test_valid_authoritative_approval_allows_iterate(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    ctx = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X),
        profile=profile,
        hitl_store=store,
        manager=CodeCraftSessionManager(),
    )
    craft_id = _open_session(ctx, tenant_id=TENANT_A, task_id=TASK_X)
    _approve(store, tenant_id=TENANT_A, task_id=TASK_X, craft_id=craft_id)

    out = codecraft_iterate(
        ctx,
        CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert out.result.error != "hitl_pending"
    assert out.result.error != "hitl_denied"


def test_codecraft_run_hitl_parity_without_approval(tmp_path: Path) -> None:
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True)
    ctx = _ctx(_sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X), profile=profile)
    with patch("intergrax.tools.providers.codecraft.service.code_exec") as mocked_exec:
        out = codecraft_run(
            ctx,
            CodeCraftRunToolInput(code="print('blocked')\n", tenant_id=TENANT_A, task_id=TASK_X),
        )
        mocked_exec.assert_not_called()
    assert out.result.error == "hitl_pending"


def test_codecraft_run_hitl_parity_with_approval(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True)
    ctx = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X),
        profile=profile,
        hitl_store=store,
    )
    craft_id = "craft-run-parity"
    _approve(store, tenant_id=TENANT_A, task_id=TASK_X, craft_id=craft_id)
    out = codecraft_run(
        ctx,
        CodeCraftRunToolInput(
            code="print('approved')\n",
            tenant_id=TENANT_A,
            task_id=TASK_X,
            craft_id=craft_id,
        ),
    )
    assert out.result.success is True
    assert "approved" in out.result.stdout


def test_non_exec_modes_regression_without_hitl(tmp_path: Path) -> None:
    dry_ctx = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X),
        profile=CodeCraftProfile(mode="dry_run"),
    )
    dry_out = codecraft_run(
        dry_ctx,
        CodeCraftRunToolInput(code="print('skip')\n", tenant_id=TENANT_A, task_id=TASK_X),
    )
    assert dry_out.result.success is True
    assert dry_out.result.stdout == ""

    assist_ctx = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_Y),
        profile=CodeCraftProfile(mode="assist_only"),
    )
    code = "print('helper')\n"
    assist_out = codecraft_run(
        assist_ctx,
        CodeCraftRunToolInput(code=code, tenant_id=TENANT_A, task_id=TASK_Y),
    )
    assert assist_out.result.success is True
    assert assist_out.result.structured_output.get("code") == code


def test_wrong_run_decision_does_not_authorize(tmp_path: Path) -> None:
    store = InMemoryHumanDecisionPersistence()
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True, require_tests=False)
    ctx = _ctx(
        _sandbox(tmp_path, tenant_id=TENANT_A, task_id=TASK_X),
        profile=profile,
        hitl_store=store,
    )
    craft_id = _open_session(ctx, tenant_id=TENANT_A, task_id=TASK_X)
    other_run = mint_run_id()
    _approve(store, tenant_id=TENANT_A, task_id=TASK_X, craft_id=craft_id, run_id=str(other_run))

    token = bind_active_execution_identity(run_id=RUN_A, attempt_id=ATTEMPT_A)
    try:
        with patch("intergrax.runtime.codecraft.orchestrator.code_exec") as mocked_exec:
            out = codecraft_iterate(
                ctx,
                CodeCraftIterateToolInput(craft_id=craft_id, tenant_id=TENANT_A, task_id=TASK_X),
            )
            mocked_exec.assert_not_called()
    finally:
        from intergrax.contracts.execution_identity import reset_active_execution_identity

        reset_active_execution_identity(token)

    assert out.result.error == "hitl_pending"
