# © Artur Czarnecki. All rights reserved.

"""Shared active execution context helpers for GraphExecutor tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)


@contextmanager
def bound_graph_execution_context(
    *,
    authority: ParentExecutionAuthority | None = None,
    bind_budget: bool = True,
) -> Iterator[None]:
    """Bind identity, authority, and optionally root budget for direct GraphExecutor calls."""
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        authority or ParentExecutionAuthority.unknown(),
    )
    budget_token = None
    if bind_budget:
        budget_token = bind_root_execution_budget(
            execution_id=execution_id,
            ledger=create_execution_budget_ledger(None),
        )
    try:
        yield
    finally:
        if budget_token is not None:
            reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
