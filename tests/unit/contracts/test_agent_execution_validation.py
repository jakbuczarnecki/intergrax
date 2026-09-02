# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.agent_execution_validation import (
    frozen_capability_validators,
)
from intergrax.contracts.validation import ValidationResult


def _validator(summary: str):
    def validate(execution: AgentExecutionResult) -> ValidationResult:
        if (execution.summary or "").strip() == summary:
            return ValidationResult(valid=True, errors=[], warnings=[])
        return ValidationResult(
            valid=False,
            errors=[f"expected summary {summary!r}"],
            warnings=[],
        )

    return validate


@pytest.mark.unit
@pytest.mark.gate
def test_frozen_capability_validators_source_mutation_isolation() -> None:
    source = {"alpha": _validator("ok")}
    registry = frozen_capability_validators(source)
    original = registry.validator_for("alpha")
    assert original is not None

    source["alpha"] = _validator("changed")
    source["beta"] = _validator("other")

    assert registry.validator_for("alpha") is original
    assert registry.validator_for("beta") is None


@pytest.mark.unit
@pytest.mark.gate
def test_frozen_capability_validators_internal_immutability() -> None:
    registry = frozen_capability_validators({"alpha": _validator("ok")})
    assert type(registry._entries) is tuple
    assert registry._entries == (("alpha", registry.validator_for("alpha")),)


@pytest.mark.unit
@pytest.mark.gate
def test_frozen_capability_validators_lookup() -> None:
    validator = _validator("ok")
    registry = frozen_capability_validators({"alpha": validator})
    assert registry.validator_for("alpha") is validator


@pytest.mark.unit
@pytest.mark.gate
def test_frozen_capability_validators_missing_capability_returns_none() -> None:
    registry = frozen_capability_validators({})
    assert registry.validator_for("missing") is None


@pytest.mark.unit
@pytest.mark.gate
def test_frozen_capability_validators_canonical_order() -> None:
    validator_a = _validator("a")
    validator_b = _validator("b")
    first = frozen_capability_validators({"beta": validator_b, "alpha": validator_a})
    second = frozen_capability_validators({"alpha": validator_a, "beta": validator_b})
    assert first == second
    assert first._entries == (("alpha", validator_a), ("beta", validator_b))


@pytest.mark.unit
@pytest.mark.gate
def test_frozen_capability_validators_empty_registry() -> None:
    registry = frozen_capability_validators({})
    assert registry.validator_for("alpha") is None
    assert registry._entries == ()
