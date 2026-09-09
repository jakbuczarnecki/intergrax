# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.execution_lineage import (
    ExecutionLineageIntegrityError,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
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
from intergrax.runtime.execution.identity_authority import mint_root_execution_identity
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget


class _SinglePersistenceChildAdmissionFailure(InMemoryExecutionLineagePersistence):
    """Fail the first child admission on the same store used for nested delegation."""

    def __init__(self) -> None:
        super().__init__()
        self._remaining_child_failures = 1

    def admit_child(self, *args: object, **kwargs: object) -> object:
        if self._remaining_child_failures > 0:
            self._remaining_child_failures -= 1
            raise ExecutionLineageUnavailableError("child admission unavailable")
        return super().admit_child(*args, **kwargs)


@pytest.mark.asyncio
async def test_nested_child_fails_closed_when_parent_admission_missing() -> None:
    persistence = _SinglePersistenceChildAdmissionFailure()
    identity = mint_root_execution_identity()
    root = identity.execution_id
    scope = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=root,
    )
    persistence.admit_root(scope, root, root)
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
    runner = ChildExecutionRunner(ledger=create_execution_budget_ledger(RunBudget()))

    class _E3Delegate:
        async def execute(self, _request: object) -> str:
            return "e3"

    class _E2Delegate:
        async def execute(self, _request: object) -> str:
            with pytest.raises(
                ExecutionLineageIntegrityError, match="parent admission missing"
            ):
                await runner.execute(request=object(), delegate=_E3Delegate())
            return "e2"

    try:
        result = await runner.execute(request=object(), delegate=_E2Delegate())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    assert result == "e2"
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    assert len(page.admissions) == 1
    assert page.admissions[0].execution_id == root
