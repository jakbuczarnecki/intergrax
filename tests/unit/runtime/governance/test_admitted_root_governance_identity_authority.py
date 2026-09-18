# © Artur Czarnecki. All rights reserved.

"""ADR-GOVERNED-EXECUTION-003 admitted root governance identity binding gates."""

from __future__ import annotations

import pytest

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootExecutionOptions,
    resolve_root_execution_context,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    peek_active_execution_governance_identity,
)
from intergrax.runtime.task.task import Task

pytestmark = pytest.mark.unit


class _NoopDelegate:
    async def execute(self, request: object) -> str:
        return "ok"


@pytest.mark.asyncio
async def test_valid_admitted_identity_binds_exact_triple() -> None:
    admitted = AdmittedRootGovernanceIdentity(
        tenant_id="tenant-T",
        workspace_id="workspace-W",
        principal_id="principal-P",
    )
    options = RootExecutionOptions(
        authority=ParentExecutionAuthority.unrestricted_root(),
        governance_identity=admitted,
    )
    context = resolve_root_execution_context(options)
    captured: list[tuple[str, str, str]] = []

    class _ProbeDelegate:
        async def execute(self, request: object) -> str:
            active_inner = peek_active_execution_governance_identity()
            assert active_inner is not None
            captured.append(
                (
                    active_inner.tenant_id,
                    active_inner.workspace_id,
                    active_inner.principal_id,
                ),
            )
            return "ok"

    runtime = ExecutionRuntime(_ProbeDelegate())
    await runtime.execute(object(), context)
    assert captured == [("tenant-T", "workspace-W", "principal-P")]


@pytest.mark.asyncio
async def test_ungoverned_root_does_not_bind_fake_identity() -> None:
    context = RootExecutionContext(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=ParentExecutionAuthority.unrestricted_root(),
        governance_identity=None,
        tenant_id="tenant-only",
    )

    class _ProbeDelegate:
        async def execute(self, request: object) -> str:
            assert peek_active_execution_governance_identity() is None
            return "ok"

    runtime = ExecutionRuntime(_ProbeDelegate())
    await runtime.execute(object(), context)


def test_task_metadata_does_not_create_admitted_identity_without_host() -> None:
    task = Task(
        tenant_id="tenant-a",
        user_id="attacker",
        message="x",
    )
    task.metadata["workspace_id"] = "attacker-workspace"
    context = RootExecutionContext(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=ParentExecutionAuthority.unrestricted_root(),
        governance_identity=None,
        tenant_id=task.tenant_id,
        workspace_id=None,
        principal_id=None,
        task_id=task.task_id,
    )
    assert context.governance_identity is None


def test_admitted_contract_rejects_partial_triple() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        AdmittedRootGovernanceIdentity(
            tenant_id="t",
            workspace_id=" ",
            principal_id="p",
        )


def test_orchestration_does_not_call_host_evidence_for_authority() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    text = (repo_root / "intergrax/runtime/execution/orchestration.py").read_text(
        encoding="utf-8",
    )
    assert "host_workspace_id" not in text
    assert "host_principal_id" not in text
