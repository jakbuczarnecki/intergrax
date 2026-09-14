# © Artur Czarnecki. All rights reserved.

"""EE-B3-A — identity spoofing and mint authority gate."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORGED_SAMPLES = (
    "not-a-canonical-id",
    "run_short",
    "exec_" + "g" * 31,
    "attempt_" + "X" * 32,
    "task_",
    "",
    "   ",
)


@pytest.mark.parametrize("forged", _FORGED_SAMPLES)
def test_ee_b3_a_forged_execution_ids_rejected(forged: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_run_id(forged)
    with pytest.raises((ValueError, TypeError)):
        validate_attempt_id(forged)
    with pytest.raises((ValueError, TypeError)):
        validate_execution_id(forged)
    with pytest.raises((ValueError, TypeError)):
        validate_task_id(forged)


def test_ee_b3_a_canonical_ids_accepted() -> None:
    suffix = "a" * 32
    validate_run_id(f"run_{suffix}")
    validate_attempt_id(f"attempt_{suffix}")
    validate_execution_id(f"exec_{suffix}")
    validate_task_id(f"task_{suffix}")


def test_ee_b3_a_ee_a2_mint_gate_still_frozen() -> None:
    from tests.unit.runtime.architecture.test_ee_a2_identity_authority_certification import (
        _collect_direct_mint_violations,
    )

    assert _collect_direct_mint_violations() == []
