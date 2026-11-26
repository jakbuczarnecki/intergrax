# © Artur Czarnecki. All rights reserved.

"""T-IGX-076-A package G: the invoker idempotency predicate stays callable without a receiver argument."""

from __future__ import annotations

import pytest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from pydantic import BaseModel

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Value(BaseModel):
    value: int


def _contract(*, side_effects: bool) -> ToolContract:
    return ToolContract(
        tool_id="probe.tool",
        name="probe.tool",
        description="probe",
        input_schema=_Value,
        output_schema=_Value,
        error_mapping={},
        side_effects=side_effects,
    )


def _request(*, idempotency_key: str | None) -> ToolExecutionRequest[BaseModel]:
    return ToolExecutionRequest(
        run_id="run-1",
        step_id="step-1",
        tool_id="probe.tool",
        input=_Value(value=1),
        idempotency_key=idempotency_key,
    )


def _receivers() -> tuple[type[RuntimeToolInvoker], RuntimeToolInvoker]:
    # The predicate and the guard use no instance state, so an uninitialised instance is enough.
    return RuntimeToolInvoker, object.__new__(RuntimeToolInvoker)


@pytest.mark.parametrize(
    ("side_effects", "idempotency_key", "expected"),
    [
        (True, "key-1", True),
        (True, None, False),
        (False, "key-1", False),
        (False, None, False),
    ],
)
def test_idempotency_predicate_via_class_and_instance(
    side_effects: bool,
    idempotency_key: str | None,
    expected: bool,
) -> None:
    contract = _contract(side_effects=side_effects)
    request = _request(idempotency_key=idempotency_key)

    for receiver in _receivers():
        assert receiver._requires_idempotency_coordination(contract, request) is expected


def test_side_effect_guard_without_coordinator_fails_closed() -> None:
    contract = _contract(side_effects=True)
    request = _request(idempotency_key="key-1")

    for receiver in _receivers():
        with pytest.raises(RuntimeError, match="requires a pre-effect coordinator"):
            receiver.assert_side_effect_tool_requires_pre_effect_coordinator(
                contract=contract,
                request=request,
                coordinator=None,
            )


@pytest.mark.parametrize(
    ("side_effects", "idempotency_key"),
    [(True, None), (False, "key-1"), (False, None)],
)
def test_side_effect_guard_is_silent_when_coordination_not_required(
    side_effects: bool,
    idempotency_key: str | None,
) -> None:
    contract = _contract(side_effects=side_effects)
    request = _request(idempotency_key=idempotency_key)

    for receiver in _receivers():
        receiver.assert_side_effect_tool_requires_pre_effect_coordinator(
            contract=contract,
            request=request,
            coordinator=None,
        )
