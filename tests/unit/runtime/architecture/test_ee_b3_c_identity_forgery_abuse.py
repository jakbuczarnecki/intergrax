# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-01 forged execution identity abuse."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    peek_active_execution_identity,
    reset_active_execution_identity,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from testing_support.security.abuse_case_fixtures import (
    CANONICAL_FORMAT_SUFFIX,
    FORGED_ID_SAMPLES,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize("forged", FORGED_ID_SAMPLES)
def test_ee_b3_c_forged_ids_fail_format_validation(forged: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_run_id(forged)
    with pytest.raises((ValueError, TypeError)):
        validate_attempt_id(forged)
    with pytest.raises((ValueError, TypeError)):
        validate_execution_id(forged)
    with pytest.raises((ValueError, TypeError)):
        validate_task_id(forged)


def test_ee_b3_c_valid_format_uuid_like_does_not_grant_active_identity() -> None:
    """Format validation alone must not mint or bind execution authority."""
    suffix = CANONICAL_FORMAT_SUFFIX
    validate_run_id(f"run_{suffix}")
    validate_execution_id(f"exec_{suffix}")
    assert peek_active_execution_identity() is None


def test_ee_b3_c_direct_mint_outside_authority_allowlist_zero() -> None:
    from tests.unit.runtime.architecture.test_ee_a2_identity_authority_certification import (
        _collect_direct_mint_violations,
    )

    assert _collect_direct_mint_violations() == []


def test_ee_b3_c_reset_active_identity_leaves_no_residual_binding() -> None:
    from intergrax.contracts.execution_identity import bind_active_execution_identity

    token = bind_active_execution_identity(
        run_id=validate_run_id(f"run_{CANONICAL_FORMAT_SUFFIX}"),
        attempt_id=validate_attempt_id(f"attempt_{CANONICAL_FORMAT_SUFFIX}"),
        execution_id=validate_execution_id(f"exec_{CANONICAL_FORMAT_SUFFIX}"),
    )
    reset_active_execution_identity(token)
    assert peek_active_execution_identity() is None
