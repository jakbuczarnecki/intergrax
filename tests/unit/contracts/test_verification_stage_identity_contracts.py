# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable

import pytest

from intergrax.contracts.domain_verification import (
    DomainVerifierId,
    validate_domain_verifier_id,
)
from intergrax.contracts.trajectory_verification import (
    TrajectoryAgentId,
    validate_trajectory_agent_id,
)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("validator", "valid_value", "expected"),
    (
        (validate_trajectory_agent_id, "agent-123", TrajectoryAgentId("agent-123")),
        (validate_domain_verifier_id, "domain.parts", DomainVerifierId("domain.parts")),
    ),
)
def test_validate_identity_accepts_valid_non_empty_string(
    validator: Callable[[object], object],
    valid_value: str,
    expected: object,
) -> None:
    assert validator(valid_value) == expected


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "validator",
    (validate_trajectory_agent_id, validate_domain_verifier_id),
)
@pytest.mark.parametrize("invalid_value", ("", "   "))
def test_validate_identity_rejects_blank_strings(
    validator: Callable[[object], object],
    invalid_value: str,
) -> None:
    with pytest.raises(ValueError):
        validator(invalid_value)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "validator",
    (validate_trajectory_agent_id, validate_domain_verifier_id),
)
@pytest.mark.parametrize("invalid_value", (None, 123, False))
def test_validate_identity_rejects_non_string_runtime_values(
    validator: Callable[[object], object],
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError):
        validator(invalid_value)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "validator",
    (validate_trajectory_agent_id, validate_domain_verifier_id),
)
def test_validate_identity_rejects_surrounding_whitespace(
    validator: Callable[[object], object],
) -> None:
    with pytest.raises(ValueError):
        validator(" leading")
