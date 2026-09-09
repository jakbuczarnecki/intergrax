# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.execution_lineage import (
    ExecutionLineageIntegrityError,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.lineage.admission import (
    build_child_lineage_admission_hook,
)
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    deactivate_root_execution_lineage,
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
    reset_active_execution_identity,
)
from intergrax.runtime.execution.identity_authority import (
    mint_child_execution_id,
    mint_root_execution_identity,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget


@pytest.mark.asyncio
async def test_nested_child_fails_closed_when_parent_admission_missing() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    identity = mint_root_execution_identity()
    root = identity.execution_id
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=root,
    )
    e2 = mint_child_execution_id()
    e3 = mint_child_execution_id()

    class _UnavailableChildPersistence(InMemoryExecutionLineagePersistence):
        def admit_child(self, *args: object, **kwargs: object) -> object:
            raise ExecutionLineageUnavailableError("child admission unavailable")

    degraded_persistence = _UnavailableChildPersistence()
    degraded_persistence.open_attempt(scope)
    degraded_persistence.open_segment(scope, root)
    degraded_persistence.admit_root(scope, root, root)

    hook = build_child_lineage_admission_hook(
        persistence=degraded_persistence,
        scope=scope,
        segment_root_execution_id=root,
        execution_id=e2,
        parent_execution_id=root,
    )
    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=root,
    )
    budget_token = bind_root_execution_budget(
        execution_id=root,
        ledger=create_execution_budget_ledger(RunBudget()),
    )

    class _E2Delegate:
        async def execute(self, _request: object) -> str:
            nested_hook = build_child_lineage_admission_hook(
                persistence=persistence,
                scope=scope,
                segment_root_execution_id=root,
                execution_id=e3,
                parent_execution_id=e2,
            )
            nested_boundary = ExecutionBoundary(
                AsyncMock(),
                admission_hooks=(nested_hook,),
                identity=ExecutionIdentityBinding(
                    run_id=identity.run_id,
                    attempt_id=identity.attempt_id,
                    execution_id=e3,
                    parent_execution_id=e2,
                ),
                authority=authority,
            )
            with pytest.raises(
                ExecutionLineageIntegrityError, match="parent admission missing"
            ):
                await nested_boundary.execute(object())
            return "e2"

    boundary = ExecutionBoundary(
        _E2Delegate(),
        admission_hooks=(hook,),
        identity=ExecutionIdentityBinding(
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            execution_id=e2,
            parent_execution_id=root,
        ),
        authority=authority,
    )
    try:
        await boundary.execute(object())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)
