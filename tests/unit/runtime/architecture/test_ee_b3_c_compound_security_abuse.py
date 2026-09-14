# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — compound security abuse (authority wins over policy; tenant+retry)."""

from __future__ import annotations

import pytest

from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    ParentExecutionAuthority,
    mint_effective_delegation_authority,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryEligibilityRequest,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.retry import (
    ExecutionAttemptRetryService,
    classify_execution_failure,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b3_c_authority_violation_wins_before_policy_allow_path() -> None:
    """Child overreach is rejected even when a permissive policy bundle might allow."""
    parent = ParentExecutionAuthority.scoped(("ops:read",))
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=parent,
            requested_permission_scopes=("ops:read", "ops:admin"),
        )


def test_ee_b3_c_wrong_tenant_retry_with_valid_run_id_no_cross_tenant_mint() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    service = ExecutionAttemptRetryService(lifecycle)
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    lifecycle.record_initial_attempt(
        tenant_id="tenant-a",
        run_id=run_id,
        attempt_id=attempt_a,
    )
    transition = service.transition_for_retry(
        tenant_id="tenant-b",
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a,
        request=ExecutionRetryEligibilityRequest(
            classification=classify_execution_failure(
                kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
            ),
            attempt_number=1,
            max_attempts=3,
        ),
    )
    assert transition is not None
    assert (
        lifecycle.get_active_attempt_id(tenant_id="tenant-a", run_id=run_id)
        == attempt_a
    )
    tenant_b_attempt = lifecycle.get_active_attempt_id(
        tenant_id="tenant-b",
        run_id=run_id,
    )
    assert tenant_b_attempt is not None
    assert tenant_b_attempt != attempt_a
