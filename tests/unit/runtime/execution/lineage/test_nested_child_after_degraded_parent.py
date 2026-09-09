# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.delegation_authority import (
    resolve_root_parent_execution_authority,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageIntegrityError,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.identity_authority import (
    mint_child_execution_id,
    mint_root_execution_identity,
)
from intergrax.runtime.execution.lineage.active_lineage import (
    peek_active_execution_lineage,
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
from intergrax.runtime.nexus.budget.budget_models import RunBudget


class _PersistentChildAdmissionOutage(InMemoryExecutionLineagePersistence):
    """Child admission always unavailable — backend stays down."""

    def __init__(self) -> None:
        super().__init__()
        self.admit_child_call_count = 0

    def admit_child(self, *args: object, **kwargs: object) -> object:
        self.admit_child_call_count += 1
        raise ExecutionLineageUnavailableError("child admission unavailable")


@pytest.mark.asyncio
async def test_nested_child_fails_closed_when_parent_lineage_non_durable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = _PersistentChildAdmissionOutage()
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
    ledger = create_execution_budget_ledger(RunBudget())
    budget_token = bind_root_execution_budget(execution_id=root, ledger=ledger)
    runner = ChildExecutionRunner(ledger=ledger)

    mint_count = 0
    original_mint = mint_child_execution_id

    def _counting_mint() -> object:
        nonlocal mint_count
        mint_count += 1
        return original_mint()

    monkeypatch.setattr(
        "intergrax.runtime.execution.child.mint_child_execution_id",
        _counting_mint,
    )

    e3_delegate_called = False
    e2_execution_id: object | None = None

    class _E3Delegate:
        async def execute(self, _request: object) -> str:
            nonlocal e3_delegate_called
            e3_delegate_called = True
            return "e3"

    class _E2Delegate:
        async def execute(self, _request: object) -> str:
            nonlocal e2_execution_id
            from intergrax.contracts.execution_identity import (
                require_active_execution_id,
            )

            e2_execution_id = require_active_execution_id()
            lineage = peek_active_execution_lineage()
            assert lineage is not None
            assert e2_execution_id in lineage.non_durable_execution_ids
            with pytest.raises(
                ExecutionLineageIntegrityError,
                match="non-durable lineage parent cannot admit nested child",
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
    assert persistence.admit_child_call_count == 1
    assert e3_delegate_called is False
    assert mint_count == 1
    assert e2_execution_id is not None
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    assert len(page.admissions) == 1
    assert page.admissions[0].execution_id == root


@pytest.mark.asyncio
async def test_sibling_from_durable_root_allowed_after_non_durable_child() -> None:
    persistence = _PersistentChildAdmissionOutage()
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
    ledger = create_execution_budget_ledger(RunBudget())
    budget_token = bind_root_execution_budget(execution_id=root, ledger=ledger)
    runner = ChildExecutionRunner(ledger=ledger)

    e2_delegate_called = False
    e3_delegate_called = False

    class _E2Delegate:
        async def execute(self, _request: object) -> str:
            nonlocal e2_delegate_called
            e2_delegate_called = True
            return "e2"

    class _E3Delegate:
        async def execute(self, _request: object) -> str:
            nonlocal e3_delegate_called
            e3_delegate_called = True
            return "e3"

    try:
        e2_result = await runner.execute(request=object(), delegate=_E2Delegate())
        e3_result = await runner.execute(request=object(), delegate=_E3Delegate())
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    assert e2_result == "e2"
    assert e3_result == "e3"
    assert e2_delegate_called is True
    assert e3_delegate_called is True
    assert persistence.admit_child_call_count == 2
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    assert len(page.admissions) == 1
    assert page.admissions[0].execution_id == root
