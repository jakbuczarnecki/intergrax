# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.errors.output_validation_error import OutputValidationError


class DummyConfig:
    production_mode = True


class DummyContext(RuntimeContext):
    pass


def test_output_validation_blocks_empty_answer():
    config = DummyConfig()
    context = RuntimeContext.__new__(RuntimeContext)
    context.config = config

    engine = RuntimeEngine(context=context)

    state = RuntimeState(
        context=context,
        request=None,
        run_id="run_test",
    )

    empty_answer = RuntimeAnswer(
        run_id="run_test",
        answer="   ",
    )

    with pytest.raises(OutputValidationError) as exc:
        engine._validate_runtime_answer_contract(
            state=state,
            runtime_answer=empty_answer,
        )

    assert exc.value.reason_code == "EMPTY_OUTPUT"


def test_output_validation_blocks_invalid_answer_type():
    config = DummyConfig()
    context = RuntimeContext.__new__(RuntimeContext)
    context.config = config

    engine = RuntimeEngine(context=context)

    state = RuntimeState(
        context=context,
        request=None,
        run_id="run_test_type",
    )

    invalid_answer = RuntimeAnswer(
        run_id="run_test_type",
        answer=123,  # intentionally invalid
    )

    with pytest.raises(OutputValidationError) as exc:
        engine._validate_runtime_answer_contract(
            state=state,
            runtime_answer=invalid_answer,
        )

    assert exc.value.reason_code == "INVALID_ANSWER_TYPE"


def test_output_validation_blocks_null_runtime_answer():
    config = DummyConfig()
    context = RuntimeContext.__new__(RuntimeContext)
    context.config = config

    engine = RuntimeEngine(context=context)

    state = RuntimeState(
        context=context,
        request=None,
        run_id="run_test_null",
    )

    with pytest.raises(OutputValidationError) as exc:
        engine._validate_runtime_answer_contract(
            state=state,
            runtime_answer=None,  # intentionally invalid
        )

    assert exc.value.reason_code == "NULL_RUNTIME_ANSWER"
