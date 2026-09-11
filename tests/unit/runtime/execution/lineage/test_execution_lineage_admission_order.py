# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_lineage import (
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.execution.identity_authority import mint_root_execution_identity
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    build_root_lineage_admission_hook,
    deactivate_root_execution_lineage,
    merge_lineage_root_admission_hooks,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.contracts.delegation_authority import (
    resolve_root_parent_execution_authority,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_task_id,
    reset_active_execution_identity,
)
from tests.unit.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt


@pytest.mark.asyncio
async def test_root_admission_before_delegate() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    identity = mint_root_execution_identity()
    task_id = mint_task_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=identity.execution_id,
    )
    order: list[str] = []

    class _FailingPersistence(InMemoryExecutionLineagePersistence):
        def admit_root(self, *args: object, **kwargs: object) -> object:
            order.append("admit_root")
            raise ExecutionLineageUnavailableError("backend down")

    failing = _FailingPersistence()
    hook = build_root_lineage_admission_hook(
        persistence=failing,
        scope=scope,
        segment_root_execution_id=identity.execution_id,
        execution_id=identity.execution_id,
    )

    async def delegate(_request: object) -> str:
        order.append("delegate")
        return "ok"

    binding = ExecutionIdentityBinding(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
    )
    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
    )
    boundary = ExecutionBoundary(
        delegate,
        admission_hooks=merge_lineage_root_admission_hooks(hook, ()),
        identity=binding,
        authority=authority,
    )
    try:
        with pytest.raises(ExecutionLineageUnavailableError):
            await boundary.execute(object())
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)
    assert order == ["admit_root"]


@pytest.mark.asyncio
async def test_child_lineage_hook_auto_attached() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    identity = mint_root_execution_identity()
    root = identity.execution_id
    task_id = mint_task_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    register_v1_attempt(persistence, scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=root,
    )
    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=root,
    )
    observed_child: list[str] = []

    class _Delegate:
        async def execute(self, _request: object) -> str:
            observed_child.append("delegate")
            return "child"

    runner = ChildExecutionRunner(ledger=create_execution_budget_ledger(RunBudget()))
    budget_token = bind_root_execution_budget(
        execution_id=root,
        ledger=create_execution_budget_ledger(RunBudget()),
    )
    try:
        await runner.execute(request=object(), delegate=_Delegate())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    child_records = [
        item for item in page.admissions if item.parent_execution_id == root
    ]
    assert len(child_records) == 1
    assert observed_child == ["delegate"]
